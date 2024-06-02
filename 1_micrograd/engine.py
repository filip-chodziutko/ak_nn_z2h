import math
from graphviz import Digraph


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f'Value(data={self.data})'

    # ---------- Binary operators ----------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def __sub__(self, other):
        return self + (-other)

    # ---------- Reversed binary ops ----------

    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    # ---------- Unary ops ----------

    def __pow__(self, other):
        assert isinstance(other, (int, float)), 'Only supporting `int` or `float` powers for now.'

        out = Value(self.data**other, _children=(self,), _op=f'**{other}')

        def _backward():
            self.grad += (other * out.data/self.data) * out.grad
        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1

    def tanh(self):
        n = self.data
        e_pow_x = math.exp(2*n)
        t = (e_pow_x - 1)/(e_pow_x + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        out = Value(math.exp(self.data), _children=(self,), _op='exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out
    
    def log(self):
        out = Value(math.log(self.data), (self,), _op='log')
        
        def _backward():
            self.grad += out.grad / self.data  # same as 1/self.data * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad = int(self.data > 0) * out.grad
        out._backward = _backward

        return out


    # ----------- Auto-grad stuff ----------
    def _get_topo(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        return topo

    def backward(self):
        topo = self._get_topo()
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label = f" {n.label} | data={n.data:.4f} | grad={n.grad:.4f} ", shape='record')
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name = uid + n._op, label = n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
