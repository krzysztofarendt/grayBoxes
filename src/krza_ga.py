import numpy as np
import matplotlib.pyplot as plt  # Needed only for the demo
import modestga as mg



def minimize(fun, x0=None, bounds=None, args=(), tol=None, method='GA',
             options={}, callback=None):
    """
    Minimize `fun` using Genetic Algorithm. Other methods might be added
    in the future.

    If `x0` is given, the initial population will contain one individual
    based on `x0`. Otherwise, all individuals will be random.

    If `bounds` are not given, assume (0, 1) for each parameter.

    `fun` arguments: `x`, `*args`.

    `callback` arguments: `x`, `fx`, `ng`, `*args`.
    `fx` is the function value at the generation `ng`.

    Return an optimization result object with the following attributes:
    - x - numpy 1D array, optimized parameters,
    - message - str, exit message,
    - ng - int, number of generations,
    - fx - float, final function value.

    Default options (can be overwritten with `options`):
    - 'generations': 100      # Max. number of generations
    - 'pop_size': 100         # Population size
    - 'mut_rate': 0.05        # Mutation rate
    - 'trm_size': 10          # Tournament size
    - 'tol': 1e-6             # Solution tolerance
    - 'inertia': 10           # Max. number of non-improving generations
    - 'xover_ratio': 0.5      # Crossover ratio

    :param fun: function to be minimized
    :param x0: numpy 1D array, initial parameters
    :param bounds: tuple, parameter bounds
    :param args: tuple, positional arguments to be passed to `fun`
    :param tol: float, solution tolerance (passed to `options`)
    :param method: str, optimization method (currently only 'GA')
    :param options: dict, GA options
    :param callback: function, called after every generation
    :return: OptRes, optimization result
    """
    # Supported optimization method
    sup_met = ['GA']

    # Parameter bounds (min and max values for each)
    # Assume min = 0, max = 1 if bounds is None
    if bounds is None:
        bounds = [(0, 1) for x in x0]

    # If tol is given, ovewrite tol from the options
    if tol is not None:
        options['tol'] = tol

    # Minimize
    if method == 'GA':
        resmg = mg.minimize(
            fun=fun, x0=x0, bounds=bounds, args=args, callback=callback,
            options=options
        )
    else:
        raise Warning(
            "Unknown method '{}'. Supported methods: {}" \
            .format(method, sup_met)
        )

    # Put results into a scipy-compatible object
    class OptimizeResult:
        """
        Optimization result with attribute names like in
        `scipy.optimize.minimize`.

        Notes:
        - success is always True (because how to define it in GA?)
        - `fun` is named `fx` in `modestga`
        """
        def __init__(self, x, fun, success, message, nfev):
            self.x = x
            self.fun = fun
            self.success = success
            self.message = message
            self.nfev = nfev

        def __str__(self):
            s = "Optimization result:\n"
            s += "====================\n"
            s += "x = {}\n".format(self.x)
            s += "success = {}\n".format(self.success)
            s += "message = {}\n".format(self.message)
            s += "fun = {}\n".format(self.fun)
            s += "nfev = {}\n".format(self.nfev)
            return s

    res = OptimizeResult(
        x=resmg.x, fun=resmg.fx, success=True, message=resmg.message,
        nfev=resmg.nfev
    )

    return res


if __name__ == "__main__":

    # DEMO
    # ===========================================

    # Function to be minimized
    def fun(x, *args):
        return np.sum(x ** 2)

    # Callback (optional)
    def callback(x, fx, ng, *args):
        """Callback function called after each generation"""

        # Print progress
        print('Generation #{}'.format(ng))
        print('    x = {}'.format(x))
        print('    fx = {}'.format(fx))

        # Save to solution history
        x_hist = args[0]
        fx_hist = args[1]
        x_hist.append(x)
        fx_hist.append(fx)

    # Parameter bounds
    bounds = [(0, 10) for i in range(10)]

    # `args` passed to `fun` and `callback`
    # Here they're used just to extract the solution history
    x_hist = list()
    fx_hist = list()
    args = (x_hist, fx_hist)

    # Additional options
    tol = 1e-6

    options = {
        'pop_size': 100,
        'trm_size': 50,
        'mut_rate': 0.1
    }

    res = minimize(
        fun=fun, x0=None, bounds=bounds, args=args, tol=tol,
        callback=callback, method='GA', options=options
    )

    # Print optimization result
    print(res)

    # Plot solution history

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(x_hist)
    ax[0].set_title('x')
    ax[1].plot(fx_hist)
    ax[1].set_title('f(x) = np.sum(x ** 2)')
    ax[1].set_xlabel('Generation')

    plt.show()
