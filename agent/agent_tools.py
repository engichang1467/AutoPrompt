from langchain.agents import tool, Tool
from langchain.pydantic_v1 import BaseModel, Field
from typing import Any, Dict, Optional, List
from copy import deepcopy
import requests
import json
import os


# data = {'dummy': 'dummy'} # A mock of the data
curr_file = os.path.dirname(__file__)
orders = json.load(open(os.path.join(curr_file, "data/retail/orders.json")))
products = json.load(open(os.path.join(curr_file, "data/retail/products.json")))
users_retail = json.load(open(os.path.join(curr_file, "data/retail/users.json")))

flights = json.load(open(os.path.join(curr_file, "data/airline/flights.json")))
reservations = json.load(open(os.path.join(curr_file, "data/airline/reservations.json")))
users_airline = json.load(open(os.path.join(curr_file, "data/airline/users.json")))


# These are the two formats of tools that can be used in the agent pipeline, you can either use @tool decorator
# or create a Tool object directly.
@tool
def magic_function(input: int) -> int:
    """Never use this tool!!"""
    return input + 1


def magic_function2(input: int) -> int:
    return input + 2

class CancelPendingOrderInput(BaseModel):
    order_id: str = Field(description="The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.")
    reason: str = Field(description="The reason for cancellation, which should be either 'no longer needed' or 'ordered by mistake'.")

@tool("cancel_pending_order", args_schema=CancelPendingOrderInput)
def cancel_pending_order(order_id: str, reason: str)-> str:
    """Cancel a pending order. If the order is already processed or delivered,
    it cannot be cancelled. The agent needs to explain the cancellation detail
    the order status will be changed to 'cancelled' and the payment will be refunded.
    The refund will be added to the user's gift card balance immediately if the payment
    was made using a gift card, otherwise the refund would take 5-7 business days to process.
    The function returns the order details after the cancellation."""

    # orders = data["orders"]
    if order_id not in orders:
        return "Error: order not found"
    order = orders[order_id]
    if order["status"] != "pending":
        return "Error: non-pending order cannot be cancelled"

    # check reason
    if reason not in ["no longer needed", "ordered by mistake"]:
        return "Error: invalid reason"

    # handle refund
    refunds = []
    for payment in order["payment_history"]:
        payment_id = payment["payment_method_id"]
        refund = {
            "transaction_type": "refund",
            "amount": payment["amount"],
            "payment_method_id": payment_id,
        }
        refunds.append(refund)
        if "gift_card" in payment_id:  # refund to gift card immediately
            # payment_method = data["users"][order["user_id"]]["payment_methods"][
            #     payment_id
            # ]
            payment_method = users_retail[order["user_id"]]["payment_methods"][
                payment_id
            ]
            payment_method["balance"] += payment["amount"]
            payment_method["balance"] = round(payment_method["balance"], 2)

    # update order status
    order["status"] = "cancelled"
    order["cancel_reason"] = reason
    order["payment_history"].extend(refunds)

    return json.dumps(order)

class Calculate(BaseModel):
    expression: str = Field(description="The mathematical expression to calculate, such as '2 + 2'. The expression can contain numbers, operators (+, -, *, /), parentheses, and spaces.")

@tool("calculate", args_schema=Calculate)
def calculate(expression: str) -> str:
    """Calculate the result of a mathematical expression."""

    if not all(char in "0123456789+-*/(). " for char in expression):
        return "Error: invalid characters in expression"
    try:
        # Evaluate the mathematical expression safely
        return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))
    except Exception as e:
        return f"Error: {e}"

class ExchangeDeliveredOrderItems(BaseModel):
    order_id: str = Field(description="The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.")
    item_ids: List[str] = Field(description="The item ids to be exchanged, each such as '1008292230'. There could be duplicate items in the list.")
    new_item_ids: List[str] = Field(description="The item ids to be exchanged for, each such as '1008292230'. There could be duplicate items in the list. Each new item id should match the item id in the same position and be of the same product.")
    payment_method_id: str = Field(description="The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'. These can be looked up from the user or order details.")

@tool("exchange_delivered_order_items", args_schema=ExchangeDeliveredOrderItems)
def exchange_delivered_order_items(order_id: str, item_ids: List[str], new_item_ids: List[str], payment_method_id: str) -> str:
    """Exchange items in a delivered order to new items of the same product type. 
    For a delivered order, return or exchange can be only done once by the agent. 
    The agent needs to explain the exchange detail and ask for explicit user confirmation (yes/no) to proceed."""
    # products, orders, users = data["products"], data["orders"], data["users"]
    users = users_retail

    # check order exists and is delivered
    if order_id not in orders:
        return "Error: order not found"
    order = orders[order_id]
    if order["status"] != "delivered":
        return "Error: non-delivered order cannot be exchanged"

    # check the items to be exchanged exist
    all_item_ids = [item["item_id"] for item in order["items"]]
    for item_id in item_ids:
        if item_ids.count(item_id) > all_item_ids.count(item_id):
            return f"Error: {item_id} not found"

    # check new items exist and match old items and are available
    if len(item_ids) != len(new_item_ids):
        return "Error: the number of items to be exchanged should match"

    diff_price = 0
    for item_id, new_item_id in zip(item_ids, new_item_ids):
        item = [item for item in order["items"] if item["item_id"] == item_id][0]
        product_id = item["product_id"]
        if not (
            new_item_id in products[product_id]["variants"]
            and products[product_id]["variants"][new_item_id]["available"]
        ):
            return f"Error: new item {new_item_id} not found or available"

        old_price = item["price"]
        new_price = products[product_id]["variants"][new_item_id]["price"]
        diff_price += new_price - old_price

    diff_price = round(diff_price, 2)

    # check payment method exists and can cover the price difference if gift card
    if payment_method_id not in users[order["user_id"]]["payment_methods"]:
        return "Error: payment method not found"

    payment_method = users[order["user_id"]]["payment_methods"][payment_method_id]
    if (
        payment_method["source"] == "gift_card"
        and payment_method["balance"] < diff_price
    ):
        return (
            "Error: insufficient gift card balance to pay for the price difference"
        )

    # modify the order
    order["status"] = "exchange requested"
    order["exchange_items"] = sorted(item_ids)
    order["exchange_new_items"] = sorted(new_item_ids)
    order["exchange_payment_method_id"] = payment_method_id
    order["exchange_price_difference"] = diff_price

    return json.dumps(order)

class FindUserIdByEmail(BaseModel):
    email: str = Field(description="The email of the user, such as 'something@example.com'.")

@tool("find_user_id_by_email", args_schema=FindUserIdByEmail)
def find_user_id_by_email(email: str) -> str:
    """Find user id by email. If the user is not found, the function will return an error message."""

    # users = data["users"]
    users = users_retail
    for user_id, profile in users.items():
        if profile["email"].lower() == email.lower():
            return user_id
    return "Error: user not found"

class FindUserIdByNameZip(BaseModel):
    first_name: str = Field(description="The first name of the customer, such as 'John'.")
    last_name: str = Field(description="The last name of the customer, such as 'Doe'.")
    zip: str = Field(description="The zip code of the customer, such as '12345'.")

@tool("find_user_id_by_name_zip", args_schema=FindUserIdByNameZip)
def find_user_id_by_name_zip(first_name: str, last_name: str, zip: str) -> str:
    """Find user id by first name, last name, and zip code. 
    If the user is not found, the function will return an error message. 
    By default, find user id by email, and only call this function if the user is not found by email or cannot remember email."""

    # users = data["users"]
    users = users_retail
    for user_id, profile in users.items():
        if (
            profile["name"]["first_name"].lower() == first_name.lower()
            and profile["name"]["last_name"].lower() == last_name.lower()
            and profile["address"]["zip"] == zip
        ):
            return user_id
    return "Error: user not found"

class GetOrderDetails(BaseModel):
    order_id: str = Field(description="The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.")

@tool("get_order_details", args_schema=GetOrderDetails)
def get_order_details(order_id: str) -> str:
    """Get the status and details of an order."""

    # orders = data["orders"]
    if order_id in orders:
        return json.dumps(orders[order_id])
    return "Error: order not found"

class GetProductDetails(BaseModel):
    product_id: str = Field(description="The product id, such as '6086499569'. Be careful the product id is different from the item id.")

@tool("get_product_details", args_schema=GetProductDetails)
def get_product_details(product_id: str) -> str:
    """Get the inventory details of a product."""

    # products = data["products"]
    if product_id in products:
        return json.dumps(products[product_id])
    return "Error: product not found"

class GetUserDetails(BaseModel):
    user_id: str = Field(description="The user id, such as 'sara_doe_496'.")

@tool("get_user_details", args_schema=GetUserDetails)
def get_user_details(user_id: str) -> str:
    """Get the details of a user."""

    # users = data["users"]
    users = users_retail
    if user_id in users:
        return json.dumps(users[user_id])
    return "Error: user not found"

class ListAllProductTypes(BaseModel):
    pass

@tool("list_all_product_types", args_schema=ListAllProductTypes)
def list_all_product_types() -> str:
    """List the name and product id of all product types. 
    Each product type has a variety of different items with unique item ids and options.
    There are only 50 product types in the store."""

    # products = data["products"]
    product_dict = {
        product["name"]: product["product_id"] for product in products.values()
    }
    product_dict = dict(sorted(product_dict.items()))
    return json.dumps(product_dict)

class ModifyPendingOrderAddress(BaseModel):
    order_id: str = Field(description="The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.")
    address1: str = Field(description="The first line of the address, such as '123 Main St'.")
    address2: str = Field(description="The second line of the address, such as 'Apt 1' or ''.")
    city: str = Field(description="The city, such as 'San Francisco'.")
    state: str = Field(description="The province, such as 'CA'.")
    country: str = Field(description="The country, such as 'USA'.")
    zip: str = Field(description="The zip code, such as '12345'.")

@tool("modify_pending_order_address", args_schema=ModifyPendingOrderAddress)
def modify_pending_order_address(
    order_id: str,
    address1: str,
    address2: str,
    city: str,
    state: str,
    country: str,
    zip: str,
) -> str:
    """Modify the shipping address of a pending order. 
    The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed."""

    # Check if the order exists and is pending
    # orders = data["orders"]
    if order_id not in orders:
        return "Error: order not found"
    order = orders[order_id]
    if order["status"] != "pending":
        return "Error: non-pending order cannot be modified"

    # Modify the address
    order["address"] = {
        "address1": address1,
        "address2": address2,
        "city": city,
        "state": state,
        "country": country,
        "zip": zip,
    }
    return json.dumps(order)

class ModifyPendingOrderItems(BaseModel):
    order_id: str = Field(description="The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.")
    item_ids: List[str] = Field(description="The item ids to be modified, each such as '1008292230'. There could be duplicate items in the list.")
    new_item_ids: List[str] = Field(description="The new item ids to be modified for, each such as '1008292230'. There could be duplicate items in the list. Each new item id should match the item id in the same position and be of the same product.")
    payment_method_id: str = Field(description="The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'. These can be looked up from the user or order details.")

@tool("modify_pending_order_items", args_schema=ModifyPendingOrderItems)
def modify_pending_order_items(order_id: str, item_ids: List[str], new_item_ids: List[str], payment_method_id: str) -> str:
    """Modify items in a pending order to new items of the same product type. 
    For a pending order, this function can only be called once. 
    The agent needs to explain the exchange detail and ask for explicit user confirmation (yes/no) to proceed."""
    
    # products, orders, users = data["products"], data["orders"], data["users"]
    users = users_retail

    # Check if the order exists and is pending
    if order_id not in orders:
        return "Error: order not found"
    order = orders[order_id]
    if order["status"] != "pending":
        return "Error: non-pending order cannot be modified"

    # Check if the items to be modified exist
    all_item_ids = [item["item_id"] for item in order["items"]]
    for item_id in item_ids:
        if item_ids.count(item_id) > all_item_ids.count(item_id):
            return f"Error: {item_id} not found"

    # Check new items exist, match old items, and are available
    if len(item_ids) != len(new_item_ids):
        return "Error: the number of items to be exchanged should match"

    diff_price = 0
    for item_id, new_item_id in zip(item_ids, new_item_ids):
        item = [item for item in order["items"] if item["item_id"] == item_id][0]
        product_id = item["product_id"]
        if not (
            new_item_id in products[product_id]["variants"]
            and products[product_id]["variants"][new_item_id]["available"]
        ):
            return f"Error: new item {new_item_id} not found or available"

        old_price = item["price"]
        new_price = products[product_id]["variants"][new_item_id]["price"]
        diff_price += new_price - old_price

    # Check if the payment method exists
    if payment_method_id not in users[order["user_id"]]["payment_methods"]:
        return "Error: payment method not found"

    # If the new item is more expensive, check if the gift card has enough balance
    payment_method = users[order["user_id"]]["payment_methods"][payment_method_id]
    if (
        payment_method["source"] == "gift_card"
        and payment_method["balance"] < diff_price
    ):
        return "Error: insufficient gift card balance to pay for the new item"

    # Handle the payment or refund
    order["payment_history"].append(
        {
            "transaction_type": "payment" if diff_price > 0 else "refund",
            "amount": abs(diff_price),
            "payment_method_id": payment_method_id,
        }
    )
    if payment_method["source"] == "gift_card":
        payment_method["balance"] -= diff_price
        payment_method["balance"] = round(payment_method["balance"], 2)

    # Modify the order
    for item_id, new_item_id in zip(item_ids, new_item_ids):
        item = [item for item in order["items"] if item["item_id"] == item_id][0]
        item["item_id"] = new_item_id
        item["price"] = products[item["product_id"]]["variants"][new_item_id][
            "price"
        ]
        item["options"] = products[item["product_id"]]["variants"][new_item_id][
            "options"
        ]
    order["status"] = "pending (item modified)"

    return json.dumps(order)

class ModifyPendingOrderPayment(BaseModel):
    order_id: str = Field(description="The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.")
    payment_method_id: str = Field(description="The payment method id to be modified, such as 'gift_card_0000000' or 'credit_card_0000000'. These can be looked up from the user or order details.")

@tool("modify_pending_order_payment", args_schema=ModifyPendingOrderPayment)
def modify_pending_order_payment(order_id: str, payment_method_id: str) -> str:
    """Modify the payment method of a pending order. 
    The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed."""

    # orders = data["orders"]

    # Check if the order exists and is pending
    if order_id not in orders:
        return "Error: order not found"
    order = orders[order_id]
    if order["status"] != "pending":
        return "Error: non-pending order cannot be modified"

    # Check if the payment method exists
    # if payment_method_id not in data["users"][order["user_id"]]["payment_methods"]:
    if payment_method_id not in users_retail[order["user_id"]]["payment_methods"]:
        return "Error: payment method not found"

    # Check that the payment history should only have one payment
    if (
        len(order["payment_history"]) > 1
        or order["payment_history"][0]["transaction_type"] != "payment"
    ):
        return "Error: there should be exactly one payment for a pending order"

    # Check that the payment method is different
    if order["payment_history"][0]["payment_method_id"] == payment_method_id:
        return (
            "Error: the new payment method should be different from the current one"
        )

    amount = order["payment_history"][0]["amount"]
    # payment_method = data["users"][order["user_id"]]["payment_methods"][
    #     payment_method_id
    # ]
    payment_method = users_retail[order["user_id"]]["payment_methods"][
        payment_method_id
    ]

    # Check if the new payment method has enough balance if it is a gift card
    if (
        payment_method["source"] == "gift_card"
        and payment_method["balance"] < amount
    ):
        return "Error: insufficient gift card balance to pay for the order"

    # Modify the payment method
    order["payment_history"].extend(
        [
            {
                "transaction_type": "payment",
                "amount": amount,
                "payment_method_id": payment_method_id,
            },
            {
                "transaction_type": "refund",
                "amount": amount,
                "payment_method_id": order["payment_history"][0][
                    "payment_method_id"
                ],
            },
        ]
    )

    # If payment is made by gift card, update the balance
    if payment_method["source"] == "gift_card":
        payment_method["balance"] -= amount
        payment_method["balance"] = round(payment_method["balance"], 2)

    # If refund is made to a gift card, update the balance
    if "gift_card" in order["payment_history"][0]["payment_method_id"]:
        old_payment_method = users_retail[order["user_id"]]["payment_methods"][
            order["payment_history"][0]["payment_method_id"]
        ]
        old_payment_method["balance"] += amount
        old_payment_method["balance"] = round(old_payment_method["balance"], 2)

    return json.dumps(order)

class ModifyUserAddress(BaseModel):
    user_id: str = Field(description="The user id, such as 'sara_doe_496'.")
    address1: str = Field(description="The first line of the address, such as '123 Main St'.")
    address2: str = Field(description="The second line of the address, such as 'Apt 1' or ''.")
    city: str = Field(description="The city, such as 'San Francisco'.")
    state: str = Field(description="The province, such as 'CA'.")
    country: str = Field(description="The country, such as 'USA'.")
    zip: str = Field(description="The zip code, such as '12345'.")

@tool("modify_user_address", args_schema=ModifyUserAddress)
def modify_user_address(
    user_id: str, address1: str, address2: str, city: str, state: str, country: str, zip: str
) -> str:
    """Modify the address of a user. 
    The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed."""

    # users = data["users"]
    users = users_retail
    if user_id not in users:
        return "Error: user not found"
    user = users[user_id]
    user["address"] = {
        "address1": address1,
        "address2": address2,
        "city": city,
        "state": state,
        "country": country,
        "zip": zip,
    }
    return json.dumps(user)

class ReturnDeliveredOrderItems(BaseModel):
    order_id: str = Field(description="The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.")
    item_ids: List[str] = Field(description="The item ids to be returned, each such as '1008292230'. There could be duplicate items in the list.")
    payment_method_id: str = Field(description="The payment method id to receive refund for the item price, such as 'gift_card_0000000' or 'credit_card_0000000'. These can be looked up from the user or order details.")

@tool("return_delivered_order_items", args_schema=ReturnDeliveredOrderItems)
def return_delivered_order_items(order_id: str, item_ids: List[str], payment_method_id: str) -> str:
    """Return items in a delivered order for a refund. 
    For a delivered order, return or exchange can be only done once by the agent. 
    The agent needs to explain the return detail and ask for explicit user confirmation (yes/no) to proceed."""

    # orders = data["orders"]

    # Check if the order exists and is delivered
    if order_id not in orders:
        return "Error: order not found"
    order = orders[order_id]
    if order["status"] != "delivered":
        return "Error: non-delivered order cannot be returned"

    # Check if the payment method exists and is either the original payment method or a gift card
    # if payment_method_id not in data["users"][order["user_id"]]["payment_methods"]:
    if payment_method_id not in users_retail[order["user_id"]]["payment_methods"]:
        return "Error: payment method not found"
    if (
        "gift_card" not in payment_method_id
        and payment_method_id != order["payment_history"][0]["payment_method_id"]
    ):
        return "Error: payment method should be either the original payment method or a gift card"

    # Check if the items to be returned exist (there could be duplicate items in either list)
    all_item_ids = [item["item_id"] for item in order["items"]]
    for item_id in item_ids:
        if item_ids.count(item_id) > all_item_ids.count(item_id):
            return "Error: some item not found"

    # Update the order status
    order["status"] = "return requested"
    order["return_items"] = sorted(item_ids)
    order["return_payment_method_id"] = payment_method_id

    return json.dumps(order)

class Think(BaseModel):
    thought: str = Field(description="A thought to think about.")

@tool("think", args_schema=Think)
def think(thought: str) -> str:
    """Use the tool to think about something. 
    It will not obtain new information or change the database, but just append the thought to the log. 
    Use it when complex reasoning or some cache memory is needed."""
    # This method does not change the state of the data; it simply returns an empty string.
    return ""

class TransferToHumanAgents(BaseModel):
    summary: str = Field(description="A summary of the user's issue.")

@tool("transfer_to_human_agents", args_schema=TransferToHumanAgents)
def transfer_to_human_agents(summary: str) -> str:
    """Transfer the user to a human agent, with a summary of the user's issue. 
    Only transfer if the user explicitly asks for a human agent, or if the user's issue cannot be resolved by the agent with the available tools."""
    # This method simulates the transfer to a human agent.
    return "Transfer successful"

# Airline
class BookReservation(BaseModel):
    user_id: str = Field(description="The ID of the user to book the reservation, such as 'sara_doe_496'.")
    origin: str = Field(description="The IATA code for the origin city, such as 'SFO'.")
    destination: str = Field(description="The IATA code for the destination city, such as 'JFK'.")
    flight_type: str = Field(description="The type of flight, either 'one-way' or 'round-trip'.")
    cabin: str = Field(description="The cabin class, either 'economy', 'premium economy', 'business', or 'first'.")
    # flights: List[Dict[str, Any]] = Field(description="The list of flights with details, such as departure and arrival times, flight number, and airline.")
    # passengers: List[Dict[str, Any]] = Field(description="The list of passengers with names and date of birth.")
    # payment_methods: List[Dict[str, Any]] = Field(description="The list of payment methods to pay for the reservation.")
    total_baggages: int = Field(description="The total number of checked and carry-on baggages for all passengers.")
    nonfree_baggages: int = Field(description="The number of non-free checked baggages for all passengers.")
    insurance: str = Field(description="The type of insurance, either 'basic', 'standard', or 'premium'.")

@tool("book_reservation", args_schema=BookReservation)
def book_reservation(
        user_id: str,
        origin: str,
        destination: str,
        flight_type: str,
        cabin: str,
        # flights: List[Dict[str, Any]],
        # passengers: List[Dict[str, Any]],
        # payment_methods: List[Dict[str, Any]],
        total_baggages: int,
        nonfree_baggages: int,
        insurance: str,
    ) -> str:
        """"""
        # reservations, users = data["reservations"], data["users"]
        users = users_airline
        # flights, passengers, payment_methods = data[""], [], []
        passengers, payment_methods = [], []
        if user_id not in users:
            return "Error: user not found"
        user = users[user_id]

        # assume each task makes at most 3 reservations
        reservation_id = "HATHAT"
        if reservation_id in reservations:
            reservation_id = "HATHAU"
            if reservation_id in reservations:
                reservation_id = "HATHAV"

        reservation = {
            "reservation_id": reservation_id,
            "user_id": user_id,
            "origin": origin,
            "destination": destination,
            "flight_type": flight_type,
            "cabin": cabin,
            "flights": deepcopy(flights),
            "passengers": passengers,
            "payment_history": payment_methods,
            "created_at": "2024-05-15T15:00:00",
            "total_baggages": total_baggages,
            "nonfree_baggages": nonfree_baggages,
            "insurance": insurance,
        }

        # update flights and calculate price
        total_price = 0
        for flight in reservation["flights"]:
            flight_number = flight["flight_number"]
            # if flight_number not in data["flights"]:
            if flight_number not in flights:
                return f"Error: flight {flight_number} not found"
            # flight_data = data["flights"][flight_number]
            flight_data = flights[flight_number]
            if flight["date"] not in flight_data["dates"]:
                return (
                    f"Error: flight {flight_number} not found on date {flight['date']}"
                )
            flight_date_data = flight_data["dates"][flight["date"]]
            if flight_date_data["status"] != "available":
                return f"Error: flight {flight_number} not available on date {flight['date']}"
            if flight_date_data["available_seats"][cabin] < len(passengers):
                return f"Error: not enough seats on flight {flight_number}"
            flight["price"] = flight_date_data["prices"][cabin]
            flight["origin"] = flight_data["origin"]
            flight["destination"] = flight_data["destination"]
            total_price += flight["price"] * len(passengers)

        if insurance == "yes":
            total_price += 30 * len(passengers)

        total_price += 50 * nonfree_baggages

        for payment_method in payment_methods:
            payment_id = payment_method["payment_id"]
            amount = payment_method["amount"]
            if payment_id not in user["payment_methods"]:
                return f"Error: payment method {payment_id} not found"
            if user["payment_methods"][payment_id]["source"] in [
                "gift_card",
                "certificate",
            ]:
                if user["payment_methods"][payment_id]["amount"] < amount:
                    return f"Error: not enough balance in payment method {payment_id}"
        if sum(payment["amount"] for payment in payment_methods) != total_price:
            return f"Error: payment amount does not add up, total price is {total_price}, but paid {sum(payment['amount'] for payment in payment_methods)}"

        # if checks pass, deduct payment and update seats
        for payment_method in payment_methods:
            payment_id = payment_method["payment_id"]
            amount = payment_method["amount"]
            if user["payment_methods"][payment_id]["source"] == "gift_card":
                user["payment_methods"][payment_id]["amount"] -= amount
            elif user["payment_methods"][payment_id]["source"] == "certificate":
                del user["payment_methods"][payment_id]

        reservations[reservation_id] = reservation
        user["reservations"].append(reservation_id)
        return json.dumps(reservation)

class CancelReservation(BaseModel):
    reservation_id: str = Field(description="The reservation ID, such as 'ZFA04Y'.")

@tool("cancel_reservation", args_schema=CancelReservation)
def cancel_reservation(reservation_id: str) -> str:
    """Cancel the whole reservation."""
    # reservations = data["reservations"]
    if reservation_id not in reservations:
        return "Error: reservation not found"
    reservation = reservations[reservation_id]

    # reverse the payment
    refunds = []
    for payment in reservation["payment_history"]:
        refunds.append(
            {
                "payment_id": payment["payment_id"],
                "amount": -payment["amount"],
            }
        )
    reservation["payment_history"].extend(refunds)
    reservation["status"] = "cancelled"
    return json.dumps(reservation)

class GetReservationDetails(BaseModel):
    reservation_id: str = Field(description="The reservation ID, such as '8JX2WO'.")

@tool("get_reservation_details", args_schema=GetReservationDetails)
def get_reservation_details(reservation_id: str) -> str:
    """Get the details of a reservation."""
    # reservations = data["reservations"]
    if reservation_id in reservations:
        return json.dumps(reservations[reservation_id])
    return "Error: user not found"

class ListAllAirports(BaseModel):
    pass

@tool("list_all_airports", args_schema=ListAllAirports)
def list_all_airports() -> str:
    """List all airports with their IATA codes."""
    airports = [
        "SFO", "JFK", "LAX", "ORD", "DFW", "DEN", "SEA", "ATL", "MIA", "BOS", "PHX", "IAH", "LAS", "MCO", "EWR", "CLT", "MSP", "DTW", "PHL", "LGA",
    ]
    cities = [
        "San Francisco", "New York", "Los Angeles", "Chicago", "Dallas", "Denver", "Seattle", "Atlanta", "Miami", "Boston", "Phoenix", "Houston", "Las Vegas", "Orlando", "Newark", "Charlotte", "Minneapolis", "Detroit", "Philadelphia", "LaGuardia",
    ]
    return json.dumps({airport: city for airport, city in zip(airports, cities)})

class SearchDirectFlight(BaseModel):
    origin: str = Field(description="The origin city airport in three letters, such as 'JFK'.")
    destination: str = Field(description="The destination city airport in three letters, such as 'LAX'.")
    date: str = Field(description="The date of the flight in the format 'YYYY-MM-DD', such as '2024-01-01'.")

@tool("search_direct_flight", args_schema=SearchDirectFlight)
def search_direct_flight(origin: str, destination: str, date: str) -> str:
    """Search direct flights between two cities on a specific date."""
    # flights = data["flights"]
    results = []
    for flight in flights.values():
        if flight["origin"] == origin and flight["destination"] == destination:
            if (
                date in flight["dates"]
                and flight["dates"][date]["status"] == "available"
            ):
                # results add flight except dates, but add flight["datas"][date]
                results.append({k: v for k, v in flight.items() if k != "dates"})
                results[-1].update(flight["dates"][date])
    return json.dumps(results)

class SearchOnestopFlight(BaseModel):
    origin: str = Field(description="The origin city airport in three letters, such as 'JFK'.")
    destination: str = Field(description="The destination city airport in three letters, such as 'LAX'.")
    date: str = Field(description="The date of the flight in the format 'YYYY-MM-DD', such as '2024-01-01'.")

@tool("search_onestop_flight", args_schema=SearchOnestopFlight)
def search_onestop_flight(origin: str, destination: str, date: str) -> str:
    """Search direct flights between two cities on a specific date."""
    # flights = data["flights"]
    results = []
    for flight1 in flights.values():
        if flight1["origin"] == origin:
            for flight2 in flights.values():
                if (
                    flight2["destination"] == destination
                    and flight1["destination"] == flight2["origin"]
                ):
                    date2 = (
                        f"2024-05-{int(date[-2:])+1}"
                        if "+1" in flight1["scheduled_arrival_time_est"]
                        else date
                    )
                    if (
                        flight1["scheduled_arrival_time_est"]
                        > flight2["scheduled_departure_time_est"]
                    ):
                        continue
                    if date in flight1["dates"] and date2 in flight2["dates"]:
                        if (
                            flight1["dates"][date]["status"] == "available"
                            and flight2["dates"][date2]["status"] == "available"
                        ):
                            result1 = {
                                k: v for k, v in flight1.items() if k != "dates"
                            }
                            result1.update(flight1["dates"][date])
                            result1["date"] = date
                            result2 = {
                                k: v for k, v in flight2.items() if k != "dates"
                            }
                            result2.update(flight2["dates"][date])
                            result2["date"] = date2
                            results.append([result1, result2])
    return json.dumps(results)

class SendCertificate(BaseModel):
    user_id: str = Field(description="The ID of the user to book the reservation, such as 'sara_doe_496'.")
    amount: float = Field(description="Certificate amount to send.")

@tool("send_certificate", args_schema=SendCertificate)
def send_certificate(user_id: str, amount: float) -> str:
    """Send a certificate to a user. Be careful!"""
    # users = data["users"]
    users = users_airline
    if user_id not in users:
        return "Error: user not found"
    user = users[user_id]

    # add a certificate, assume at most 3 cases per task
    for id in [3221322, 3221323, 3221324]:
        payment_id = f"certificate_{id}"
        if payment_id not in user["payment_methods"]:
            user["payment_methods"][payment_id] = {
                "source": "certificate",
                "amount": amount,
                "id": payment_id,
            }
            return f"Certificate {payment_id} added to user {user_id} with amount {amount}."
    
class UpdateReservationBaggages(BaseModel):
    reservation_id: str = Field(description="The reservation ID, such as 'ZFA04Y'.")
    total_baggages: int = Field(description="The updated total number of baggage items included in the reservation.")
    nonfree_baggages: int = Field(description="The updated number of non-free baggage items included in the reservation.")
    payment_id: str = Field(description="The payment id stored in user profile, such as 'credit_card_7815826', 'gift_card_7815826', 'certificate_7815826'.")

@tool("update_reservation_baggages", args_schema=UpdateReservationBaggages)
def update_reservation_baggages(reservation_id: str, total_baggages: int, nonfree_baggages: int, payment_id: str) -> str:
    """Update the number of baggage items in a reservation."""
    # users, reservations = data["users"], data["reservations"]
    users = users_airline
    if reservation_id not in reservations:
        return "Error: reservation not found"
    reservation = reservations[reservation_id]

    total_price = 50 * max(0, nonfree_baggages - reservation["nonfree_baggages"])
    if payment_id not in users[reservation["user_id"]]["payment_methods"]:
        return "Error: payment method not found"
    payment_method = users[reservation["user_id"]]["payment_methods"][payment_id]
    if payment_method["source"] == "certificate":
        return "Error: certificate cannot be used to update reservation"
    elif (
        payment_method["source"] == "gift_card"
        and payment_method["amount"] < total_price
    ):
        return "Error: gift card balance is not enough"

    reservation["total_baggages"] = total_baggages
    reservation["nonfree_baggages"] = nonfree_baggages
    if payment_method["source"] == "gift_card":
        payment_method["amount"] -= total_price

    if total_price != 0:
        reservation["payment_history"].append(
            {
                "payment_id": payment_id,
                "amount": total_price,
            }
        )

    return json.dumps(reservation)

class UpdateReservationFlights(BaseModel):
    reservation_id: str = Field(description="The reservation ID, such as 'ZFA04Y'.")
    cabin: str = Field(description="The updated cabin class, either 'basic_economy', 'economy', or 'business'.")
    # An array of objects containing details about each piece of flight in the ENTIRE new reservation. Even if the a flight segment is not changed, it should still be included in the array.
    flights: List[Dict[str, Any]] = Field(description="An array of objects containing details about each piece of flight in the ENTIRE new reservation. Even if the a flight segment is not changed, it should still be included in the array.")
    payment_id: str = Field(description="The payment id stored in user profile, such as 'credit_card_7815826', 'gift_card_7815826', 'certificate_7815826'.")

@tool("update_reservation_flights", args_schema=UpdateReservationFlights)
def update_reservation_flights(reservation_id: str, cabin: str, flights: List[Dict[str, Any]], payment_id: str) -> str:
    """Update the flights in a reservation."""
    # users, reservations = data["users"], data["reservations"]
    users = users_airline
    if reservation_id not in reservations:
        return "Error: reservation not found"
    reservation = reservations[reservation_id]

    # update flights and calculate price
    total_price = 0
    flights = deepcopy(flights)
    for flight in flights:
        # if existing flight, ignore
        if _ := [
            f
            for f in reservation["flights"]
            if f["flight_number"] == flight["flight_number"]
            and f["date"] == flight["date"]
            and cabin == reservation["cabin"]
        ]:
            total_price += _[0]["price"] * len(reservation["passengers"])
            flight["price"] = _[0]["price"]
            flight["origin"] = _[0]["origin"]
            flight["destination"] = _[0]["destination"]
            continue
        flight_number = flight["flight_number"]
        # if flight_number not in data["flights"]:
        if flight_number not in flights:
            return f"Error: flight {flight_number} not found"
        # flight_data = data["flights"][flight_number]
        flight_data = flights[flight_number]
        if flight["date"] not in flight_data["dates"]:
            return (
                f"Error: flight {flight_number} not found on date {flight['date']}"
            )
        flight_date_data = flight_data["dates"][flight["date"]]
        if flight_date_data["status"] != "available":
            return f"Error: flight {flight_number} not available on date {flight['date']}"
        if flight_date_data["available_seats"][cabin] < len(
            reservation["passengers"]
        ):
            return f"Error: not enough seats on flight {flight_number}"
        flight["price"] = flight_date_data["prices"][cabin]
        flight["origin"] = flight_data["origin"]
        flight["destination"] = flight_data["destination"]
        total_price += flight["price"] * len(reservation["passengers"])

    total_price -= sum(flight["price"] for flight in reservation["flights"]) * len(
        reservation["passengers"]
    )

    # check payment
    if payment_id not in users[reservation["user_id"]]["payment_methods"]:
        return "Error: payment method not found"
    payment_method = users[reservation["user_id"]]["payment_methods"][payment_id]
    if payment_method["source"] == "certificate":
        return "Error: certificate cannot be used to update reservation"
    elif (
        payment_method["source"] == "gift_card"
        and payment_method["amount"] < total_price
    ):
        return "Error: gift card balance is not enough"

    # if checks pass, deduct payment and update seats
    if payment_method["source"] == "gift_card":
        payment_method["amount"] -= total_price
    reservation["flights"] = flights
    if total_price != 0:
        reservation["payment_history"].append(
            {
                "payment_id": payment_id,
                "amount": total_price,
            }
        )
    # do not make flight database update here, assume it takes time to be updated
    return json.dumps(reservation)

class UpdateReservationPassengers(BaseModel):
    reservation_id: str = Field(description="The reservation ID, such as 'ZFA04Y'.")
    passengers: List[Dict[str, Any]] = Field(description="The updated list of passengers with names and date of birth.")

@tool("update_reservation_passengers", args_schema=UpdateReservationPassengers)
def update_reservation_passengers(reservation_id: str, passengers: List[Dict[str, Any]]) -> str:
    """Update the passengers in a reservation."""
    # reservations = data["reservations"]
    if reservation_id not in reservations:
        return "Error: reservation not found"
    reservation = reservations[reservation_id]
    if len(passengers) != len(reservation["passengers"]):
        return "Error: number of passengers does not match"
    reservation["passengers"] = passengers
    return json.dumps(reservation)


@tool
def parse_yaml_code(yaml_code: str) -> str:
    """You must use this tool before sending the final output, the input is the yaml code with the output schema. The result is the final output!"""
    return "The Yaml doesn't have a valid yaml structure, please fix it such that it can be parsed. Remember that if you have a value that is a string, you should wrap it in quotes."
