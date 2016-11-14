from ucf_exchange_client import Strategy, Buy, Sell, Cancel

import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


def orders_from_target(target, book):
	"""return the orders necessary to achieve the price target."""
    orders = list() 
    current_price = .5*(book.bids[-1].price + book.asks[0].price)

    if current_price == target:
        return orders
    elif current_price < target:
        # need to make book.min_ask so that current_price' is target
        target_min_ask = 2*target - book.bids[-1].price
        filter_below = lambda ask: ask.price < target_min_ask
        relevant_asks = filter(filter_below, book.asks)
        for ask in relevant_asks:
            orders.add(Buy(ask.price, ask.quantity))  

        orders.add(Sell(target_min_ask, book.bids[-1].quantity))
    else: #current_price > target
        target_max_bid = 2*target - book.asks[0].price
        filter_above = lambda bid: bid.price > target_max_bid
        relevant_bids = filter(filter_above, book.bids)
        for bid in relevant_bids:
            orders.add(Sell(bid.price, bid.quantity))

        orders.add(Buy(target_max_bid, book.asks[0].price))
        
    return orders


def model_from_price_trajectory(samples):
    models = dict()
    for asset in samples:
        data = samples[asset]
        X = np.array([data[i][0] for i in range(len(data))]).reshape(-1, 1)
        y = np.array([data[i][1] for i in range(len(data))]).reshape(-1, 1)
        models[asset] = GaussianProcessRegressor(kernel=Matern(nu=2.5)).fit(X, y)
    return models


def plot_gpr(gpr):
    # assuming that X is increasing
    X = gpr.X_train_.reshape(-1, 1)
    y = gpr.y_train_.reshape(-1, 1)

    x_fill = np.linspace(X[0, 0], X[-1, 0], 1000).reshape(-1, 1)
    y_pred, sigma = gpr.predict(x_fill, return_std=True)
    
    y_pred = y_pred.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)

    fig = plt.figure()
    plt.scatter(X, y, label='observations')
    plt.plot(x_fill, y_pred, 'b-', label='prediction')
    plt.plot(x_fill, gpr.sample_y(x_fill).reshape(-1, 1), 'g')
    upper, lower = y_pred + 1.96*sigma, y_pred - 1.96*sigma
    plt.fill_between(x_fill.squeeze(), upper.squeeze(), lower.squeeze(), color='r', alpha='0.2')
    plt.xlabel('time')
    plt.ylabel('price')
    plt.legend(loc='upper left')

    plt.show()


def adjust_prices(my_model):
	invisible_hand = Strategy('invisible_hand')

	@invisible_hand.handle('Hello')
	def init(state, msg):
		state.gp = my_model
		return []

	@invisible_hand.handle('Book')
	def push_market(state, msg):
		# something like this
		time = state.clock.now()
		for asset in msg.keys():
			price_target = state.gp.sample(time)
			# yield from ... in python 3.
			for order in orders_from_target(price_target, msg[asset]):
				yield order
