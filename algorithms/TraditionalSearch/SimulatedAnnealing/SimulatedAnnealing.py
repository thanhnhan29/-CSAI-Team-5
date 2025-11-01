import numpy as np
from typing import Callable, Union


class SimulatedAnnealing:
    def __init__(
        self,
        func: Callable[[Union[float, np.ndarray]], float],
        T0: float = 5.0,
        alpha: float = 0.98,
        max_iter: int = 1000,
        step_size: float = 0.1,
        mode: Union[str, Callable[[float, float], bool]] = "min",
        seed: int | None = None
    ):
        """
        Parameters
        ----------
        func : Callable
            Hàm mục tiêu f(x) cần tối ưu. x có thể là float hoặc np.ndarray.
        T0 : float
            Nhiệt độ ban đầu.
        alpha : float
            Hệ số giảm nhiệt độ theo hàm mũ.
        max_iter : int
            Số vòng lặp tối đa.
        step_size : float
            Biên độ bước nhảy để sinh nghiệm lân cận.
        mode : str hoặc Callable
            - "min" : tối thiểu hoá.
            - "max" : tối đa hoá.
            - Callable(old, new) : hàm so sánh tuỳ chỉnh, trả về True nếu new tốt hơn old.
        """
        if not (isinstance(mode, str) or callable(mode)):
            raise ValueError("mode phải là 'min', 'max' hoặc một hàm so sánh (callable).")

        if isinstance(mode, str) and mode not in ("min", "max"):
            raise ValueError("mode phải là 'min' hoặc 'max'.")
        
        if seed is not None:
            np.random.seed(seed)

        self.func = func
        self.T = T0
        self.alpha = alpha
        self.max_iter = max_iter
        self.step_size = step_size
        self.mode = mode

    @staticmethod
    def random_neighbor(x: np.ndarray, step_size: float = 0.1) -> np.ndarray:
        return x + np.random.uniform(-step_size, step_size, size=x.shape)

    @staticmethod
    def exponential_cooling(T: float, alpha: float) -> float:
        return alpha * T

    def _is_better(self, old: float, new: float) -> bool:
        if callable(self.mode):
            return self.mode(old, new)
        elif self.mode == "min":
            return new < old
        elif self.mode == "max":
            return new > old
        return False

    def optimize(
        self,
        x0: Union[float, tuple, list, np.ndarray],
        return_history: bool = False
    ) -> Union[
        tuple[np.ndarray, float],
        tuple[np.ndarray, float, list[tuple[np.ndarray, float]]]
    ]:
        """
        Chạy tối ưu hoá bằng Simulated Annealing.

        Parameters
        ----------
        x0 : float, tuple, list hoặc np.ndarray
            Nghiệm khởi tạo.
        return_history : bool, optional
            Nếu True, trả thêm lịch sử [(x_t, f_t), ...].

        Returns
        -------
        tuple
            (x_best, f(x_best)) nếu return_history=False,
            hoặc (x_best, f(x_best), history) nếu return_history=True.
        """
        x = np.array(x0, dtype=float)
        fx = self.func(*x) if x.ndim > 0 else self.func(x)

        x_best, f_best = x.copy(), fx
        history = [x.copy()]

        for _ in range(self.max_iter):
            x_new = self.random_neighbor(x, self.step_size)
            f_new = self.func(*x_new) if x_new.ndim > 0 else self.func(x_new)
            delta = f_new - fx

            if self._is_better(fx, f_new) or np.random.rand() < np.exp(-abs(delta) / self.T):
                x, fx = x_new, f_new
                if self._is_better(f_best, f_new):
                    x_best, f_best = x_new.copy(), f_new

            self.T = self.exponential_cooling(self.T, self.alpha)
            history.append(x.copy())

        if x_best.size == 1:
            x_best = float(x_best[0])

        return (x_best, f_best, history) if return_history else (x_best, f_best)