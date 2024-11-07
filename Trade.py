from dataclasses import dataclass, field
from datetime import datetime as dt
from time import time
from decimal import Decimal

"""
TradeInfo class for storing trade information.

:param orderId: The order ID.
:type orderId: int
:example: 123456789

:param type: The order type.
:type type: str
:example: 'limit' or 'market'

:param orderType: The order type.
:type orderType: str
:example: 'open' or 'close'

:param timestamp: The timestamp of the trade.
:type timestamp: int
:example: 1583140140

:param datetime: The datetime of the trade.
:type datetime: datetime
:example: datetime.datetime(2020, 3, 2, 0, 0)

:param symbol: The symbol of the stock.
:type symbol: str
:example: 'AAPL.US'

:param size: The size of the trade.
:type size: int
:example: 1000

:param price: The price of the trade.
:type price: Decimal
:example: 100.0

:param amount: The amount of the trade. (THIS IS POSITION VALUE)
:type amount: Decimal
:example: 100000.0 (size * price)

:param side: The side of the trade.
:type side: str
:example: 'buy' or 'sell'

:param fee: The fee of the trade.
:type fee: Decimal

:param total: The total of the trade.
:type total: Decimal
:example: 100000.0 (amount + fee)

:param stopPrice: The stop price of the trade.
:type stopPrice: Decimal

:param closePosition: Whether the trade is a close position trade.
:type closePosition: bool

:param realizedPnl: The realized PnL of the trade.
:type realizedPnl: list

:param realizedPnlRate: The realized PnL rate of the trade.
:type realizedPnlRate: list

:param income: The income of the trade. (THIS IS NET INCOME)
:type income: Decimal

:param incomeRate: The income rate of the trade.
:type incomeRate: Decimal

:param totalIncome: The total income of the trade.
:type totalIncome: Decimal

:param totalIncomeRate: The total income rate of the trade.
:type totalIncomeRate: Decimal
"""

@dataclass
class TradeInfo:
    orderId: int
    type: str
    orderType: str
    createTime: int
    datetime: dt
    symbol: str
    size: Decimal
    price: Decimal
    amount: Decimal
    side: str
    fee: Decimal
    total: Decimal
    updateTime: int = None
    isStop: bool = False
    stopPrice: dict = field(default_factory=dict)
    closePosition: bool = False
    realizedPnl: list = field(default_factory=list)
    realizedPnlRate: list = field(default_factory=list)
    income: Decimal = None
    incomeRate: Decimal = None
    totalIncome: Decimal = None
    totalIncomeRate: Decimal = None
    cost: Decimal = None

    def to_dict(self):
        return {
            'orderId': int(self.orderId),
            'type': self.type,
            'orderType': self.orderType,
            'createTime': int(self.createTime),
            'datetime': self.datetime.strftime("%Y-%m-%d %H:%M:%S"),
            'symbol': self.symbol,
            'size': Decimal(self.size),
            'price': Decimal(self.price),
            'amount': Decimal(self.amount),
            'side': self.side,
            'fee': Decimal(self.fee),
            'total': Decimal(self.total),
            'updateTime': int(self.updateTime) if self.updateTime is not None else None,
            'isStop': self.isStop,
            'stopPrice': self.stopPrice,
            'closePosition': self.closePosition,
            'realizedPnl': self.realizedPnl,
            'realizedPnlRate': self.realizedPnlRate,
            'income': Decimal(self.income) if self.income is not None else None,
            'incomeRate': Decimal(self.incomeRate) if self.incomeRate is not None else None,
            'totalIncome': Decimal(self.totalIncome) if self.totalIncome is not None else None,
            'totalIncomeRate': Decimal(self.totalIncomeRate) if self.totalIncomeRate is not None else None
        }
    
class Trade:
    pass

class FakeTrade:
    def __init__(self) -> None:
        self.open_trade_records = {}
        self.close_trade_records = {}
        self.trade_records = {}

    def reset(self):
        self.open_trade_records = {}
        self.close_trade_records = {}
        self.trade_records = {}

    def get_open_trade_records(self, symbol: str = None) -> dict:
        return self.open_trade_records.get(symbol, self.open_trade_records)
    
    def get_close_trade_records(self, symbol: str = None) -> dict:
        return self.close_trade_records.get(symbol, self.close_trade_records)
    
    def get_trade_records(self, symbol: str = None) -> dict:
        return self.trade_records.get(symbol, self.trade_records)

    def open_order(self, symbol: str, size: Decimal, side: str, price: Decimal, datetime: dt, stop_price: Decimal = None) -> TradeInfo:
        """
        Open a market order.

        :param symbol: The symbol of the stock.
        :type symbol: str

        :param size: The size of the trade.
        :type size: Decimal

        :param side: The side of the trade.
        :type side: str

        :param price: The price of the trade.
        :type price: Decimal

        :param stop_price: The stop price of the trade.
        :type stop_price: Decimal
        :default: None

        :return: TradeInfo
        """
        if size <= 0:
            raise ValueError("Size must be bigger than 0.")
        
        if side not in ['buy', 'sell']:
            raise ValueError("Side must be 'buy' or 'sell'.")
        
        if price <= 0:
            raise ValueError("Price must be bigger than 0.")
        
        if stop_price is not None:
            if stop_price <= 0:
                raise ValueError("Stop price must be bigger than 0.")
            if (side == "buy" and stop_price >= price) or (side == "sell" and stop_price <= price):
                print(symbol, side, stop_price, price)
                raise ValueError("Stop price is not valid.")
        
        order_id = int(round(time() * 1000))
        order_type = 'fake'
        trade_type = 'open'
        timestamp = int(round(time() * 1000))
        now = datetime
        amount = size * price
        fee = Decimal('0.0')
        total = amount + fee
        _stop_price = {size: stop_price} if stop_price is not None else None
        
        trade_info = TradeInfo(
                    orderId=order_id,
                    type=order_type,
                    orderType=trade_type, 
                    createTime=timestamp, 
                    datetime=now, 
                    symbol=symbol, 
                    size=size,
                    price=price,
                    amount=amount,
                    side=side, 
                    fee=fee,
                    total=total,
                    stopPrice=_stop_price
                )
        
        self.open_trade_records.setdefault(symbol, {})[order_id] = trade_info
        self.trade_records.setdefault(symbol, []).append(trade_info)

        return trade_info
    
    def increase_position(self, symbol: str, order_id: int, size: Decimal, price: Decimal, stop_price: Decimal = None) -> TradeInfo:
        """
        Increase the trade position.

        :param symbol: The symbol of the stock.
        :type symbol: str

        :param order_id: The order id of the trade.
        :type order_id: int

        :param size: The size of the trade.
        :type size: Decimal

        :param price: The price of the trade.
        :type price: Decimal

        :param stop_price: The stop price of the trade.
        :type stop_price: Decimal
        :default: None

        :return: TradeInfo
        """

        if size <= 0:
            raise ValueError("Size must be bigger than 0.")
        
        if price <= 0:
            raise ValueError("Price must be bigger than 0.")
        
        _records = self.open_trade_records.get(symbol, None)

        if _records is None:
            raise ValueError("This symbol does not have any open trade records.")
        
        _trade_info = _records.get(order_id, None)

        if _trade_info is None:
            raise ValueError("This order id does not exist.")
        
        if stop_price is not None:
            if stop_price <= 0:
                raise ValueError("Stop price must be bigger than 0.")
            if (_trade_info.side == "buy" and stop_price >= price) or (_trade_info.side == "sell" and stop_price <= price):
                raise ValueError("Stop price is not valid.")
            
        old_amount = self.open_trade_records[symbol][order_id].amount
        old_size = self.open_trade_records[symbol][order_id].size
        
        _amount = size * price
        _fee = Decimal('0.0')
        _total = _amount + _fee

        new_amount = old_amount + _amount

        _price = new_amount / (old_size + size)

        self.open_trade_records[symbol][order_id].price = _price
        self.open_trade_records[symbol][order_id].size += size
        self.open_trade_records[symbol][order_id].amount += _amount
        self.open_trade_records[symbol][order_id].fee += _fee
        self.open_trade_records[symbol][order_id].total += _total

        if stop_price is not None:
            self.open_trade_records[symbol][order_id].stopPrice[size] = stop_price

        self.open_trade_records[symbol][order_id].updateTime = int(round(time() * 1000))

        return self.open_trade_records[symbol][order_id]

    ### need to be fixed
    def close_order(self, symbol: str, order_id: int, price: Decimal, size: Decimal, datetime:dt, is_stop:bool=False) -> TradeInfo:
        """
        Close a order by orderId.

        :param symbol: The symbol of the stock.
        :type symbol: str

        :param order_id: The order ID.
        :type order_id: Decimal

        :param price: The price of the trade.
        :type price: Decimal

        :param size: The size of the trade.
        :type size: int

        :param datetime: The datetime of the trade.
        :type datetime: datetime

        :param is_stop: Whether the trade is a stop trade.
        :type is_stop: bool
        :default: False

        :return: TradeInfo
        """

        temp_trade_records = self.open_trade_records.get(symbol, None)

        if temp_trade_records is None:
            raise Exception("No open order for symbol: {}".format(symbol))
        
        open_trade_info = temp_trade_records.get(order_id)

        if open_trade_info is None:
            raise Exception("No open order for order_id: {}".format(order_id))
        
        if price <= 0:
            raise ValueError("Price can't be less than 0. Your price: {}, symbol: {}".format(price, symbol))
        
        if size <= 0:
            raise ValueError("Size can't be less than 0")
        
        open_trade_size = open_trade_info.size
        open_trade_price = open_trade_info.price
        open_trade_total = open_trade_info.total
        open_trade_side = open_trade_info.side
        open_trade_realized_pnl = open_trade_info.realizedPnl

        # size can't not bigger than open_trade_info.size
        if size > open_trade_info.size:
            size = open_trade_info.size

        amount = size * price
        open_price_amount = size * open_trade_price
        less_size = open_trade_size - size
        less_amount = less_size * open_trade_price

        # calculate the income and income rate
        if open_trade_side == "buy":
            income = amount - open_price_amount
        else:
            income = open_price_amount - amount

        income_rate = income / open_price_amount
            
        # close the order
        close_order_id = int(round(time() * 1000))
        order_type = 'fake'
        trade_type = 'close'
        timestamp = int(round(time() * 1000))
        now = datetime
        size = size
        amount = amount
        side = "sell" if open_trade_info.side == "buy" else "buy"
        fee = Decimal('0.0')
        total = amount + fee
        close_position = True if less_size == 0 else False
        total_income = None
        total_income_rate = None

        if close_position:
            total_income = sum(open_trade_realized_pnl) + income
            total_income_rate = total_income / open_trade_total

        close_trade_info = TradeInfo(
                            orderId=close_order_id,
                            type=order_type,
                            orderType=trade_type,
                            createTime=timestamp,
                            datetime=now,
                            symbol=symbol,
                            size=size,
                            price=price,
                            amount=amount,
                            side=side,
                            fee=fee,
                            total=total,
                            closePosition=close_position,
                            income=income,
                            incomeRate=income_rate,
                            totalIncome=total_income,
                            totalIncomeRate=total_income_rate,
                            isStop=is_stop,
                            cost=open_trade_info.total
                        )
        
        # records the close order info
        self.close_trade_records.setdefault(symbol, {})[close_order_id] = close_trade_info
        self.trade_records[symbol].append(close_trade_info)

        # change the open trade info
        if less_size != 0:
            # if size is not equal to open trade size, change the open trade info
            # records realized profit
            self.open_trade_records[symbol][order_id].size = less_size
            self.open_trade_records[symbol][order_id].amount = less_amount
            self.open_trade_records[symbol][order_id].realizedPnl.append(income)
            self.open_trade_records[symbol][order_id].realizedPnlRate.append(income_rate)
        else:
            # if size is equal to open trade size, pop the open trade info
            self.open_trade_records[symbol].pop(order_id)
            self.open_trade_records.pop(symbol)

        return close_trade_info, open_trade_price, open_trade_info.datetime

    def close_all_order(self, symbol: str, price: Decimal, datetime:dt) -> list:
        """
        Close all orders.

        :param symbol: The symbol of the stock.
        :type symbol: str

        :param price: The price of the trade.
        :type price: Decimal

        :return: list [TradeInfo]
        """

        open_trade_records = self.open_trade_records.get(symbol, None)

        if open_trade_records is None or open_trade_records == {}:
            raise Exception("No open order for symbol: {}".format(symbol))
        
        if price <= 0:
            raise ValueError("Price can't be less than 0")
        
        close_orders = []
        
        for _, open_trade_info in open_trade_records.items():
            amount = open_trade_info.size * price
            income = amount - open_trade_info.amount if open_trade_info.side == "buy" else open_trade_info.amount - amount
            income_rate = income / open_trade_info.amount
            close_order_id = int(round(time() * 1000))
            order_type = 'fake'
            trade_type = 'close'
            timestamp = int(round(time() * 1000))
            now = datetime
            size = open_trade_info.size
            amount = amount
            side = "sell" if open_trade_info.side == "buy" else "buy"
            fee = Decimal('0.0')
            total = amount + fee
            close_position = True
            total_income = sum(open_trade_info.realizedPnl) + income
            total_income_rate = total_income / open_trade_info.total

            close_trade_info = TradeInfo(
                                orderId=close_order_id,
                                type=order_type,
                                orderType=trade_type,
                                createTime=timestamp,
                                datetime=now,
                                symbol=symbol,
                                size=size,
                                price=price,
                                amount=amount,
                                side=side,
                                fee=fee,
                                total=total,
                                closePosition=close_position,
                                income=income,
                                incomeRate=income_rate,
                                totalIncome=total_income,
                                totalIncomeRate=total_income_rate,
                                cost=open_trade_info.total
                            )
            
            self.close_trade_records.setdefault(symbol, {})[close_order_id] = close_trade_info
            self.trade_records[symbol].append(close_trade_info)

            close_orders.append(close_trade_info)

        self.open_trade_records[symbol].clear()
        self.open_trade_records.pop(symbol)

        return close_orders
    
if __name__ == "__main__":
    # test
    fake_trade = FakeTrade()
    open_order = fake_trade.open_order(
        symbol="AAPL",
        price=100.0,
        size=1000,
        side="buy",
        datetime=dt.now()
    )
    print(open_order)
    fake_trade.increase_position(
        symbol="AAPL",
        order_id=open_order.orderId,
        price=90.0,
        size=1000,
    )
    close_order = fake_trade.close_order(
        symbol="AAPL",
        order_id=open_order.orderId,
        price=125.0,
        size=250,
        datetime=dt.now()
    )
    print(close_order)
    data = fake_trade.get_open_trade_records('AAPL')
    print(data)

    close_order = fake_trade.close_all_order(
        symbol="AAPL",
        price=147.0,
        datetime=dt.now()
    )
    print(close_order)

    data = fake_trade.get_open_trade_records('AAPL')
    print(data)
    pass