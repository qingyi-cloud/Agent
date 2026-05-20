"""规则打分型司机决策服务。

该实现只依赖评测环境注入的 ``SimulationApiPort``。决策时通过接口获取状态、
历史和候选货源，使用确定性规则完成过滤与打分，避免每步都消耗模型 token。
"""

from __future__ import annotations

import logging
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from simkit.ports import SimulationApiPort


_SIMULATION_EPOCH = datetime(2026, 3, 1, 0, 0, 0)
_WALL_TIME_FMT = "%Y-%m-%d %H:%M:%S"
_DEFAULT_SPEED_KM_PER_HOUR = 60.0
_DEFAULT_COST_PER_KM = 1.5
_DEFAULT_WAIT_MINUTES = 30
_MIN_WAIT_MINUTES = 15
_MONTH_HORIZON_MINUTES = 30 * 24 * 60
_MODEL_CANDIDATE_LIMIT = 6


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_km = 6371.0
    p1 = math.radians(lat1)
    l1 = math.radians(lng1)
    p2 = math.radians(lat2)
    l2 = math.radians(lng2)
    dp = p2 - p1
    dl = l2 - l1
    h = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(dl * 0.5) ** 2)
    h = min(1.0, max(0.0, h))
    return 2.0 * radius_km * math.asin(math.sqrt(h))


def _distance_to_minutes(distance_km: float, speed_km_per_hour: float = _DEFAULT_SPEED_KM_PER_HOUR) -> int:
    if distance_km <= 0:
        return 1
    return max(1, math.ceil((distance_km / speed_km_per_hour) * 60))


def _wall_time_to_minutes(text: str) -> int | None:
    try:
        dt = datetime.strptime(str(text).strip(), _WALL_TIME_FMT)
    except (TypeError, ValueError):
        return None
    return int((dt - _SIMULATION_EPOCH).total_seconds() // 60)


def _parse_number(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def _intervals_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def _preference_text(preference: Any) -> str:
    if isinstance(preference, str):
        return preference.strip()
    if isinstance(preference, dict):
        return str(preference.get("content") or preference.get("text") or "").strip()
    return ""


def _preference_penalty(preference: Any) -> float:
    if not isinstance(preference, dict):
        return 0.0
    try:
        return float(preference.get("penalty_amount", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class NightWindow:
    start_minute: int
    end_minute: int

    def wait_minutes_if_active(self, now_minutes: int) -> int:
        day = now_minutes // 1440
        minute_of_day = now_minutes % 1440
        if self.start_minute <= self.end_minute:
            start_abs = day * 1440 + self.start_minute
            end_abs = day * 1440 + self.end_minute
            if start_abs <= now_minutes < end_abs:
                return end_abs - now_minutes
            return 0

        if minute_of_day >= self.start_minute:
            end_abs = (day + 1) * 1440 + self.end_minute
            return end_abs - now_minutes
        if minute_of_day < self.end_minute:
            end_abs = day * 1440 + self.end_minute
            return end_abs - now_minutes
        return 0

    def overlaps(self, start_minutes: int, end_minutes: int) -> bool:
        for day in range(start_minutes // 1440 - 1, end_minutes // 1440 + 2):
            if self.start_minute <= self.end_minute:
                window_start = day * 1440 + self.start_minute
                window_end = day * 1440 + self.end_minute
            else:
                window_start = day * 1440 + self.start_minute
                window_end = (day + 1) * 1440 + self.end_minute
            if _intervals_overlap(start_minutes, end_minutes, window_start, window_end):
                return True
        return False


@dataclass(frozen=True)
class GeoBounds:
    lat_min: float
    lat_max: float
    lng_min: float
    lng_max: float

    def contains(self, lat: float, lng: float) -> bool:
        return self.lat_min <= lat <= self.lat_max and self.lng_min <= lng <= self.lng_max


@dataclass
class PreferenceProfile:
    hard_forbidden_categories: set[str] = field(default_factory=set)
    soft_forbidden_categories: dict[str, float] = field(default_factory=dict)
    night_windows: list[NightWindow] = field(default_factory=list)
    daily_rest_minutes: int = 0
    max_pickup_km: float | None = None
    max_haul_km: float | None = None
    required_cargo_ids: set[str] = field(default_factory=set)
    geo_bounds: GeoBounds | None = None

    @classmethod
    def from_preferences(cls, preferences: list[Any]) -> "PreferenceProfile":
        profile = cls()
        for preference in preferences:
            text = _preference_text(preference)
            if not text:
                continue
            profile._parse_categories(text, _preference_penalty(preference))
            profile._parse_rest(text)
            profile._parse_night_windows(text)
            profile._parse_distance_limits(text)
            profile._parse_required_cargo(text)
            profile._parse_geo_bounds(text)
        return profile

    def _parse_categories(self, text: str, penalty: float) -> None:
        categories = set(re.findall(r"「([^」]+)」", text))
        if not categories:
            return
        if "尽量" in text and ("不拉" in text or "不接" in text):
            for category in categories:
                self.soft_forbidden_categories[category] = max(
                    penalty,
                    self.soft_forbidden_categories.get(category, 0.0),
                )
            return
        if "不接" in text or "禁接" in text:
            self.hard_forbidden_categories.update(categories)

    def _parse_rest(self, text: str) -> None:
        if "休息" not in text and "歇" not in text and "停车" not in text:
            return
        if "连续" not in text and "连着" not in text:
            return
        match = re.search(r"(\d+(?:\.\d+)?)\s*小时", text)
        if not match:
            return
        minutes = int(float(match.group(1)) * 60)
        self.daily_rest_minutes = max(self.daily_rest_minutes, minutes)

    def _parse_night_windows(self, text: str) -> None:
        if "不接单" not in text or ("不空" not in text and "空车" not in text):
            return
        for start, end in re.findall(r"(\d{1,2})点至次日(?:早)?(\d{1,2})点", text):
            self.night_windows.append(NightWindow(int(start) * 60, int(end) * 60))
        for start, end in re.findall(r"凌晨(\d{1,2})点至(\d{1,2})点", text):
            self.night_windows.append(NightWindow(int(start) * 60, int(end) * 60))
        for start, end in re.findall(r"(\d{1,2})点至(?:下午)?(\d{1,2})点", text):
            start_h = int(start)
            end_h = int(end)
            if start_h == 0 or end_h == 0:
                continue
            if start_h == 23 or end_h <= 6 or "凌晨" in text or "中午" in text:
                self.night_windows.append(NightWindow(start_h * 60, end_h * 60))

    def _parse_distance_limits(self, text: str) -> None:
        if "不得超过" not in text and "不超过" not in text and "≤" not in text:
            return
        value = _parse_number(text)
        if value is None:
            return
        if "赴装货点" in text or "空驶距离" in text:
            self.max_pickup_km = value if self.max_pickup_km is None else min(self.max_pickup_km, value)
        elif "单笔" in text and ("距离" in text or "装卸" in text):
            self.max_haul_km = value if self.max_haul_km is None else min(self.max_haul_km, value)

    def _parse_required_cargo(self, text: str) -> None:
        for cargo_id in re.findall(r"(?:编号|货源编号)\s*(\d+)", text):
            self.required_cargo_ids.add(cargo_id)

    def _parse_geo_bounds(self, text: str) -> None:
        lat_match = re.search(r"北纬\s*(\d+(?:\.\d+)?)\s*至\s*(\d+(?:\.\d+)?)", text)
        lng_match = re.search(r"东经\s*(\d+(?:\.\d+)?)\s*至\s*(\d+(?:\.\d+)?)", text)
        if not lat_match or not lng_match:
            return
        lat_a, lat_b = float(lat_match.group(1)), float(lat_match.group(2))
        lng_a, lng_b = float(lng_match.group(1)), float(lng_match.group(2))
        self.geo_bounds = GeoBounds(
            lat_min=min(lat_a, lat_b),
            lat_max=max(lat_a, lat_b),
            lng_min=min(lng_a, lng_b),
            lng_max=max(lng_a, lng_b),
        )

    def restricted_wait_minutes(self, now_minutes: int) -> int:
        waits = [window.wait_minutes_if_active(now_minutes) for window in self.night_windows]
        waits = [minutes for minutes in waits if minutes > 0]
        return max(waits) if waits else 0

    def overlaps_restricted_window(self, start_minutes: int, end_minutes: int) -> bool:
        return any(window.overlaps(start_minutes, end_minutes) for window in self.night_windows)

    def allows_position(self, lat: float, lng: float) -> bool:
        return self.geo_bounds is None or self.geo_bounds.contains(lat, lng)


@dataclass
class CargoOption:
    cargo_id: str
    cargo_name: str
    score: float
    net_income: float
    finish_minutes: int
    total_minutes: int
    pickup_km: float
    haul_km: float
    wait_minutes: int
    price: float


class ModelDecisionService:
    """单步决策入口：规则过滤 + 收益/时间打分。"""

    def __init__(
        self,
        api: SimulationApiPort,
        *,
        use_model: bool = True,
        model_candidate_limit: int = _MODEL_CANDIDATE_LIMIT,
    ) -> None:
        self._api = api
        self._use_model = use_model
        self._model_candidate_limit = max(1, int(model_candidate_limit))
        self._logger = logging.getLogger("agent.decision_service")

    def decide(self, driver_id: str) -> dict[str, Any]:
        status = self._api.get_driver_status(driver_id)
        profile = PreferenceProfile.from_preferences(list(status.get("preferences") or []))
        now_minutes = int(status.get("simulation_progress_minutes", 0) or 0)

        restricted_wait = profile.restricted_wait_minutes(now_minutes)
        if restricted_wait > 0:
            return self._wait_action(restricted_wait)

        lat = float(status["current_lat"])
        lng = float(status["current_lng"])
        self._safe_query_history(driver_id)
        cargo_resp = self._api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng)
        action_status = self._api.get_driver_status(driver_id)
        action_now = int(action_status.get("simulation_progress_minutes", now_minutes) or now_minutes)

        items = cargo_resp.get("items", [])
        if not isinstance(items, list):
            items = []
        options = [
            option
            for option in (
                self._analyze_cargo_item(
                    item=item,
                    status=action_status,
                    profile=profile,
                    now_minutes=action_now,
                )
                for item in items
            )
            if option is not None
        ]

        if options:
            best = max(options, key=lambda option: option.score)
            model_action = self._model_rerank_action(
                driver_id=driver_id,
                status=action_status,
                options=options,
                fallback=best,
            )
            if model_action is not None:
                return model_action
            self._logger.info(
                "rule decision take_order driver_id=%s cargo_id=%s score=%.2f net=%.2f finish_min=%s",
                driver_id,
                best.cargo_id,
                best.score,
                best.net_income,
                best.finish_minutes,
            )
            return {"action": "take_order", "params": {"cargo_id": best.cargo_id}}

        return self._wait_action(self._fallback_wait_minutes(profile, action_now))

    def _model_rerank_action(
        self,
        *,
        driver_id: str,
        status: dict[str, Any],
        options: list[CargoOption],
        fallback: CargoOption,
    ) -> dict[str, Any] | None:
        if not self._use_model:
            return None

        candidates = sorted(options, key=lambda option: option.score, reverse=True)[: self._model_candidate_limit]
        candidate_ids = {option.cargo_id for option in candidates}
        payload = self._build_model_payload(driver_id, status, candidates)
        try:
            response = self._api.model_chat_completion(payload)
        except Exception:
            self._logger.warning("model rerank failed; fallback to rule decision", exc_info=True)
            return None

        decision = self._extract_model_decision(response)
        action_name = str(decision.get("action", "")).strip()
        cargo_id = str(decision.get("cargo_id", "") or decision.get("id", "")).strip()
        if action_name == "take_order" and cargo_id in candidate_ids:
            self._logger.info(
                "model decision take_order driver_id=%s cargo_id=%s fallback_cargo_id=%s",
                driver_id,
                cargo_id,
                fallback.cargo_id,
            )
            return {"action": "take_order", "params": {"cargo_id": cargo_id}}

        self._logger.info(
            "model decision invalid action=%s cargo_id=%s; fallback_cargo_id=%s",
            action_name,
            cargo_id,
            fallback.cargo_id,
        )
        return None

    def _build_model_payload(
        self,
        driver_id: str,
        status: dict[str, Any],
        candidates: list[CargoOption],
    ) -> dict[str, Any]:
        candidate_payload = [
            {
                "cargo_id": option.cargo_id,
                "cargo_name": option.cargo_name,
                "rule_score": round(option.score, 2),
                "estimated_net_income": round(option.net_income, 2),
                "price": round(option.price, 2),
                "total_minutes": option.total_minutes,
                "pickup_km": round(option.pickup_km, 2),
                "haul_km": round(option.haul_km, 2),
                "wait_minutes": option.wait_minutes,
                "finish_minute": option.finish_minutes,
            }
            for option in candidates
        ]
        preferences = [_preference_text(pref) for pref in list(status.get("preferences") or [])]
        user_payload = {
            "driver_id": driver_id,
            "simulation_progress_minutes": int(status.get("simulation_progress_minutes", 0) or 0),
            "current_position": {
                "lat": status.get("current_lat"),
                "lng": status.get("current_lng"),
            },
            "preferences": [text for text in preferences if text],
            "candidate_cargos": candidate_payload,
            "instruction": "Only choose one cargo_id from candidate_cargos. Return JSON only.",
        }
        return {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a truck dispatch decision assistant. Hard constraints have already been "
                        "filtered by code. Choose the candidate with the best balance of income, time, "
                        "low empty driving, and preference safety. Reply only with compact JSON like "
                        "{\"action\":\"take_order\",\"cargo_id\":\"123456\",\"reason\":\"...\"}."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":")),
                },
            ],
            "temperature": 0.1,
            "max_tokens": 96,
            "enable_thinking": False,
        }

    def _extract_model_decision(self, response: dict[str, Any]) -> dict[str, Any]:
        content = ""
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = str(message.get("content") or "")
                if not content:
                    content = str(first.get("text") or "")
        if not content:
            content = str(response.get("content") or "")
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content).strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.S)
            if not match:
                return {}
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
        return parsed if isinstance(parsed, dict) else {}

    def _safe_query_history(self, driver_id: str) -> None:
        try:
            self._api.query_decision_history(driver_id, 5)
        except Exception:
            self._logger.debug("query_decision_history unavailable", exc_info=True)

    def _analyze_cargo_item(
        self,
        *,
        item: dict[str, Any],
        status: dict[str, Any],
        profile: PreferenceProfile,
        now_minutes: int,
    ) -> CargoOption | None:
        cargo = item.get("cargo")
        if not isinstance(cargo, dict):
            return None

        cargo_id = str(cargo.get("cargo_id", "")).strip()
        if not cargo_id:
            return None
        cargo_name = str(cargo.get("cargo_name", "") or "").strip()
        if cargo_name in profile.hard_forbidden_categories and cargo_id not in profile.required_cargo_ids:
            return None

        if not self._truck_length_matches(status.get("truck_length"), cargo.get("truck_length")):
            return None

        start = cargo.get("start") or {}
        end = cargo.get("end") or {}
        try:
            start_lat = float(start["lat"])
            start_lng = float(start["lng"])
            end_lat = float(end["lat"])
            end_lng = float(end["lng"])
            duration_minutes = int(cargo.get("cost_time_minutes") or 0)
            price = float(cargo.get("price") or 0.0)
        except (KeyError, TypeError, ValueError):
            return None
        if duration_minutes <= 0 or price <= 0:
            return None
        create_minutes = _wall_time_to_minutes(str(cargo.get("create_time", "")))
        remove_minutes = _wall_time_to_minutes(str(cargo.get("remove_time", "")))
        if create_minutes is not None and create_minutes > now_minutes:
            return None
        if remove_minutes is not None and remove_minutes < now_minutes:
            return None

        current_lat = float(status["current_lat"])
        current_lng = float(status["current_lng"])
        if not (
            profile.allows_position(current_lat, current_lng)
            and profile.allows_position(start_lat, start_lng)
            and profile.allows_position(end_lat, end_lng)
        ):
            return None
        pickup_km = self._pickup_distance_km(item, current_lat, current_lng, start_lat, start_lng)
        haul_km = _haversine_km(start_lat, start_lng, end_lat, end_lng)

        if profile.max_pickup_km is not None and pickup_km > profile.max_pickup_km:
            return None
        if profile.max_haul_km is not None and haul_km > profile.max_haul_km:
            return None

        pickup_minutes = 0 if pickup_km <= 1e-6 else _distance_to_minutes(pickup_km)
        arrival_minutes = now_minutes + pickup_minutes
        load_window = self._load_window_minutes(cargo)
        wait_minutes = 0
        if load_window is not None:
            load_start, load_end = load_window
            if arrival_minutes > load_end:
                return None
            wait_minutes = max(0, load_start - arrival_minutes)

        total_minutes = pickup_minutes + wait_minutes + duration_minutes
        finish_minutes = now_minutes + total_minutes
        if profile.overlaps_restricted_window(now_minutes, finish_minutes):
            return None
        if finish_minutes > _MONTH_HORIZON_MINUTES:
            return None

        cost_per_km = self._cost_per_km(status)
        net_income = price - (pickup_km + haul_km) * cost_per_km
        if net_income <= 0 and cargo_id not in profile.required_cargo_ids:
            return None

        hours = max(total_minutes / 60.0, 0.25)
        soft_penalty = profile.soft_forbidden_categories.get(cargo_name, 0.0)
        required_bonus = 10_000.0 if cargo_id in profile.required_cargo_ids else 0.0
        score = (net_income / hours) - soft_penalty - wait_minutes * 0.2 - pickup_km * 0.5 + required_bonus
        return CargoOption(
            cargo_id=cargo_id,
            cargo_name=cargo_name,
            score=score,
            net_income=net_income,
            finish_minutes=finish_minutes,
            total_minutes=total_minutes,
            pickup_km=pickup_km,
            haul_km=haul_km,
            wait_minutes=wait_minutes,
            price=price,
        )

    def _pickup_distance_km(
        self,
        item: dict[str, Any],
        current_lat: float,
        current_lng: float,
        start_lat: float,
        start_lng: float,
    ) -> float:
        try:
            distance = float(item.get("distance_km"))
            if distance >= 0:
                return distance
        except (TypeError, ValueError):
            pass
        return _haversine_km(current_lat, current_lng, start_lat, start_lng)

    def _load_window_minutes(self, cargo: dict[str, Any]) -> tuple[int, int] | None:
        raw = cargo.get("load_time")
        if not isinstance(raw, list) or len(raw) != 2:
            return None
        start = _wall_time_to_minutes(str(raw[0]))
        end = _wall_time_to_minutes(str(raw[1]))
        if start is None or end is None or end < start:
            return None
        return start, end

    def _truck_length_matches(self, driver_length: Any, cargo_lengths: Any) -> bool:
        if not cargo_lengths:
            return True
        if not isinstance(cargo_lengths, list):
            return True
        driver = str(driver_length or "").strip()
        if not driver:
            return True
        return driver in {str(item).strip() for item in cargo_lengths}

    def _cost_per_km(self, status: dict[str, Any]) -> float:
        try:
            cost = float(status.get("cost_per_km", _DEFAULT_COST_PER_KM))
        except (TypeError, ValueError):
            return _DEFAULT_COST_PER_KM
        return cost if cost > 0 else _DEFAULT_COST_PER_KM

    def _fallback_wait_minutes(self, profile: PreferenceProfile, now_minutes: int) -> int:
        restricted_wait = profile.restricted_wait_minutes(now_minutes)
        if restricted_wait > 0:
            return restricted_wait
        if profile.daily_rest_minutes > 0 and now_minutes % 1440 >= 21 * 60:
            return max(profile.daily_rest_minutes, _MIN_WAIT_MINUTES)
        return _DEFAULT_WAIT_MINUTES

    def _wait_action(self, duration_minutes: int) -> dict[str, Any]:
        duration = max(_MIN_WAIT_MINUTES, int(duration_minutes))
        return {"action": "wait", "params": {"duration_minutes": duration}}
