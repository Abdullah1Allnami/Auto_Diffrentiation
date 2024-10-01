class Scalar:
    def __init__(self, value, _parents=(), _operations=''):
        self.value = value
        self.gradient = 0
        self._backpropagation = lambda: None  # Empty lambda for later override
        self._ancestors = set(_parents)
        self._operations = _operations

    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        result = Scalar(self.value * other.value, (self, other), '*')

        def _backpropagation():
            self.gradient += other.value * result.gradient
            other.gradient += self.value * result.gradient

        result._backpropagation = _backpropagation
        return result

    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        result = Scalar(self.value + other.value, (self, other), '+')

        def _backpropagation():
            self.gradient += result.gradient
            other.gradient += result.gradient

        result._backpropagation = _backpropagation
        return result

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        result = Scalar(self.value ** other, (self,), f'**{other}')

        def _backpropagation():
            self.gradient += other * (self.value ** (other - 1)) * result.gradient

        result._backpropagation = _backpropagation
        return result

    def relu(self):
        result = Scalar(0 if self.value < 0 else self.value, (self,), "ReLU")

        def _backpropagation():
            self.gradient += (1 if result.value > 0 else 0) * result.gradient

        result._backpropagation = _backpropagation
        return result

    def backpropagate(self):
        # Perform a topological sort to traverse nodes in reverse order
        order = []
        visited = set()

        def build_order(node):
            if node not in visited:
                visited.add(node)
                for parent in node._ancestors:
                    build_order(parent)
                order.append(node)

        build_order(self)

        # Reset gradients before starting backpropagation
        self.gradient = 1
        for node in reversed(order):
            node._backpropagation()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def __repr__(self):
        return f"Scalar(value={self.value}, gradient={self.gradient})"
