from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from typing import Any


DEMO_ROOT = Path(__file__).resolve().parents[1]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

simkit_stub = types.ModuleType("simkit")
ports_stub = types.ModuleType("simkit.ports")
ports_stub.SimulationApiPort = object
sys.modules.setdefault("simkit", simkit_stub)
sys.modules.setdefault("simkit.ports", ports_stub)

from agent.model_decision_service import ModelDecisionService


def cargo_item(
    cargo_id: str,
    *,
    price: float,
    distance_km: float,
    cargo_name: str = "配件零件",
    haul_minutes: int = 180,
    start_lat: float = 22.5,
    start_lng: float = 113.9,
    end_lat: float = 23.0,
    end_lng: float = 113.8,
    truck_lengths: list[str] | None = None,
    create_time: str = "2026-03-01 00:00:00",
    remove_time: str = "2026-03-31 23:59:00",
) -> dict[str, Any]:
    return {
        "distance_km": distance_km,
        "cargo": {
            "cargo_id": cargo_id,
            "cargo_name": cargo_name,
            "create_time": create_time,
            "remove_time": remove_time,
            "price": price,
            "cost_time_minutes": haul_minutes,
            "load_time": ["2026-03-01 00:00:00", "2026-03-31 23:59:00"],
            "start": {"lat": start_lat, "lng": start_lng},
            "end": {"lat": end_lat, "lng": end_lng},
            "truck_length": truck_lengths or ["4.2米"],
        },
    }


class FakeApi:
    def __init__(
        self,
        *,
        status: dict[str, Any] | None = None,
        items: list[dict[str, Any]] | None = None,
        history: list[dict[str, Any]] | None = None,
        model_response: dict[str, Any] | None = None,
        query_scan_minutes: int = 0,
    ) -> None:
        self.status = {
            "driver_id": "D001",
            "current_lat": 22.54,
            "current_lng": 114.06,
            "truck_length": "4.2米",
            "preferences": [],
            "simulation_progress_minutes": 8 * 60,
            "simulation_wall_time": "2026-03-01 08:00:00",
            "completed_order_count": 0,
        }
        if status:
            self.status.update(status)
        self.items = items or []
        self.history = history or []
        self.model_response = model_response
        self.query_scan_minutes = query_scan_minutes
        self.model_payloads: list[dict[str, Any]] = []
        self.model_calls = 0
        self.query_points: list[tuple[float, float]] = []

    def get_driver_status(self, driver_id: str) -> dict[str, Any]:
        return dict(self.status, driver_id=driver_id)

    def query_cargo(self, driver_id: str, latitude: float, longitude: float) -> dict[str, Any]:
        self.query_points.append((latitude, longitude))
        self.status["simulation_progress_minutes"] = int(self.status.get("simulation_progress_minutes", 0)) + self.query_scan_minutes
        return {"driver_id": driver_id, "items": list(self.items)}

    def query_decision_history(self, driver_id: str, step: int) -> dict[str, Any]:
        return {
            "driver_id": driver_id,
            "total_steps": len(self.history),
            "returned_count": len(self.history),
            "records": list(self.history),
        }

    def model_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.model_calls += 1
        self.model_payloads.append(payload)
        if self.model_response is None:
            raise AssertionError("deterministic agent should not call the model for standard decisions")
        return self.model_response


class RuleBasedDecisionServiceTest(unittest.TestCase):
    def test_selects_best_net_income_per_hour_without_model_call(self) -> None:
        api = FakeApi(
            items=[
                cargo_item("near_low", price=180.0, distance_km=5.0, haul_minutes=120),
                cargo_item("far_high", price=900.0, distance_km=40.0, haul_minutes=180),
            ]
        )

        action = ModelDecisionService(api, use_model=False).decide("D001")

        self.assertEqual(action, {"action": "take_order", "params": {"cargo_id": "far_high"}})
        self.assertEqual(api.model_calls, 0)

    def test_filters_hard_forbidden_cargo_category(self) -> None:
        api = FakeApi(
            status={
                "preferences": [
                    {
                        "content": "不接货源品类为「蔬菜」的订单。",
                        "penalty_amount": 350,
                        "penalty_cap": 3500,
                    }
                ]
            },
            items=[
                cargo_item("veg", price=2000.0, distance_km=4.0, cargo_name="蔬菜", haul_minutes=120),
                cargo_item("safe", price=250.0, distance_km=6.0, cargo_name="配件零件", haul_minutes=120),
            ],
        )

        action = ModelDecisionService(api, use_model=False).decide("D002")

        self.assertEqual(action, {"action": "take_order", "params": {"cargo_id": "safe"}})
        self.assertEqual(api.model_calls, 0)

    def test_waits_through_night_restriction_before_querying_cargo(self) -> None:
        api = FakeApi(
            status={
                "simulation_progress_minutes": 23 * 60 + 30,
                "simulation_wall_time": "2026-03-01 23:30:00",
                "preferences": [
                    {
                        "content": "每晚23点至次日早6点不接单、不空车赶路。",
                        "penalty_amount": 200,
                        "penalty_cap": 6000,
                    }
                ],
            },
            items=[cargo_item("tempting", price=2000.0, distance_km=2.0)],
        )

        action = ModelDecisionService(api, use_model=False).decide("D005")

        self.assertEqual(action, {"action": "wait", "params": {"duration_minutes": 390}})
        self.assertEqual(api.query_points, [])
        self.assertEqual(api.model_calls, 0)

    def test_uses_wait_when_no_viable_cargo_exists(self) -> None:
        api = FakeApi(items=[])

        action = ModelDecisionService(api, use_model=False).decide("D001")

        self.assertEqual(action["action"], "wait")
        self.assertGreaterEqual(action["params"]["duration_minutes"], 15)
        self.assertEqual(api.model_calls, 0)

    def test_filters_cargo_that_would_overlap_restricted_window(self) -> None:
        api = FakeApi(
            status={
                "simulation_progress_minutes": 22 * 60,
                "simulation_wall_time": "2026-03-01 22:00:00",
                "preferences": [
                    {
                        "content": "每晚23点至次日早6点不接单、不空车赶路。",
                        "penalty_amount": 200,
                        "penalty_cap": 6000,
                    }
                ],
            },
            items=[cargo_item("crosses_night", price=2000.0, distance_km=0.0, haul_minutes=180)],
        )

        action = ModelDecisionService(api, use_model=False).decide("D005")

        self.assertEqual(action["action"], "wait")
        self.assertEqual(api.model_calls, 0)

    def test_respects_geographic_bounds_preference(self) -> None:
        api = FakeApi(
            status={
                "preferences": [
                    {
                        "content": "我就在深圳干活，不出市；跑车或停车时，车辆位置须始终在深圳市范围内（北纬22.42至22.89，东经113.74至114.66）。",
                        "penalty_amount": 2000,
                        "penalty_cap": 2000,
                    }
                ]
            },
            items=[
                cargo_item(
                    "outside",
                    price=5000.0,
                    distance_km=3.0,
                    start_lat=22.6,
                    start_lng=114.0,
                    end_lat=23.2,
                    end_lng=113.9,
                    haul_minutes=120,
                ),
                cargo_item(
                    "inside",
                    price=300.0,
                    distance_km=4.0,
                    start_lat=22.6,
                    start_lng=114.0,
                    end_lat=22.7,
                    end_lng=114.1,
                    haul_minutes=120,
                ),
            ],
        )

        action = ModelDecisionService(api, use_model=False).decide("D001")

        self.assertEqual(action, {"action": "take_order", "params": {"cargo_id": "inside"}})
        self.assertEqual(api.model_calls, 0)

    def test_model_can_rerank_viable_candidates(self) -> None:
        api = FakeApi(
            items=[
                cargo_item("near_low", price=180.0, distance_km=5.0, haul_minutes=120),
                cargo_item("far_high", price=900.0, distance_km=40.0, haul_minutes=180),
            ],
            model_response={
                "choices": [
                    {
                        "message": {
                            "content": '{"action":"take_order","cargo_id":"near_low","reason":"shorter and safer"}'
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

        action = ModelDecisionService(api, use_model=True).decide("D001")

        self.assertEqual(action, {"action": "take_order", "params": {"cargo_id": "near_low"}})
        self.assertEqual(api.model_calls, 1)
        self.assertIn("messages", api.model_payloads[0])
        self.assertIs(api.model_payloads[0].get("enable_thinking"), False)

    def test_invalid_model_choice_falls_back_to_rule_best(self) -> None:
        api = FakeApi(
            items=[
                cargo_item("near_low", price=180.0, distance_km=5.0, haul_minutes=120),
                cargo_item("far_high", price=900.0, distance_km=40.0, haul_minutes=180),
            ],
            model_response={
                "choices": [{"message": {"content": '{"action":"take_order","cargo_id":"not_visible"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

        action = ModelDecisionService(api, use_model=True).decide("D001")

        self.assertEqual(action, {"action": "take_order", "params": {"cargo_id": "far_high"}})
        self.assertEqual(api.model_calls, 1)

    def test_filters_cargo_expired_after_query_scan_cost(self) -> None:
        api = FakeApi(
            status={
                "simulation_progress_minutes": 8 * 60,
                "simulation_wall_time": "2026-03-01 08:00:00",
            },
            query_scan_minutes=10,
            items=[
                cargo_item(
                    "expired_after_scan",
                    price=2000.0,
                    distance_km=2.0,
                    haul_minutes=120,
                    remove_time="2026-03-01 08:05:00",
                ),
                cargo_item("still_online", price=300.0, distance_km=6.0, haul_minutes=120),
            ],
        )

        action = ModelDecisionService(api, use_model=False).decide("D001")

        self.assertEqual(action, {"action": "take_order", "params": {"cargo_id": "still_online"}})


if __name__ == "__main__":
    unittest.main()
