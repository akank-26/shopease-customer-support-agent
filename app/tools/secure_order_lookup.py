
import json
import re
from pathlib import Path

from app.tools.order_lookup import get_order_status


ROOT = Path(__file__).resolve().parents[2]
OWNERS_FILE = ROOT / "data" / "order_owners.json"


class OrderServiceUnavailable(Exception):
    """Raised when the order service is unavailable."""


def lookup_order_for_customer(
    order_id: str,
    authenticated_customer_id: str,
) -> dict:
    """Look up an order only if the customer owns it."""

    if not authenticated_customer_id:
        return {"error": "Authentication required"}

    if not isinstance(order_id, str):
        return {"error": "Invalid order ID"}

    order_id = order_id.strip().upper()

    if not re.fullmatch(r"ORD-\d{4}", order_id):
        return {"error": "Invalid order ID"}

    try:
        with OWNERS_FILE.open(encoding="utf-8") as file:
            owners = json.load(file)

        if not isinstance(owners, dict):
            raise ValueError("Invalid ownership data")

    except (OSError, ValueError, TypeError) as exc:
        raise OrderServiceUnavailable(
            "Ownership data unavailable"
        ) from exc

    if owners.get(order_id) != authenticated_customer_id:
        return {"error": "Order unavailable"}

    try:
        return get_order_status.invoke({
            "order_id": order_id
        })
    except (OSError, ValueError, TypeError) as exc:
        raise OrderServiceUnavailable(
            "Order lookup unavailable"
        ) from exc
