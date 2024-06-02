import math

import torch

from engine import Value

x1 = 3.5
x2 = 2.0
k = 2
val1 = Value(x1)
val2 = Value(x2)

def compare_results(res1, res2, eps=1e-6):
     return abs(res1 - res2) < eps

def test_binary_ops():
    binary_ops = (
        (Value.__add__, x1 + x2),
        (Value.__sub__, x1 - x2),
        (Value.__mul__, x1 * x2),
        (Value.__truediv__, x1 / x2)
    )

    for op, result in binary_ops:
        assert compare_results(op(val1, val2).data, result), f'{op.__name__.replace("_", "")} error given 2 Value objects'
        assert compare_results(op(val1, x2).data, result), f'{op.__name__.replace("_", "")} error given Value and float'

def test_reverserd_binary_ops():
        msg = "{op} error given float and Value"
        assert compare_results((x1 + val2).data, (x1 + x2)), msg.format(op='add')
        assert compare_results((x1 - val2).data, (x1 - x2)), msg.format(op='sub')
        assert compare_results((x1 * val2).data, (x1 * x2)), msg.format(op='mul')

def test_unary_ops():
    assert compare_results((-val1).data, (-x1)), 'neg error'
    assert compare_results((val1**k).data, (x1**k)), 'pow error'
    assert compare_results((val1.tanh()).data, (math.tanh(x1))), 'tanh error'
    assert compare_results((val1.exp()).data, (math.exp(x1))), 'exp error'
    assert compare_results((val1.log()).data, (math.log(x1))), 'log error'
    assert compare_results((val1.relu()).data, (max(0, x1))), 'relu error'

def test_backward():
    ten1 = torch.Tensor([x1])
    ten2 = torch.Tensor([x2])

    ten1.requires_grad = True
    ten2.requires_grad = True

    test_cases = (
         ((val1 + val2), (ten1 + ten2), 'add'),
         ((val1 - val2), (ten1 - ten2), 'sub'),
         ((val1 * val2), (ten1 * ten2), 'mul'),
         ((val1 / val2), (ten1 / ten2), 'div'),
         ((val1**k), (ten1**k), 'pow'),
         ((val1.tanh()), (ten1.tanh()), 'tanh'),
         ((val1.exp()), (ten1.exp()), 'exp'),
         ((val1.log()), (ten1.log()), 'log'),
         ((val1.relu()), (ten1.relu()), 'relu'),
    )

    for mcgrd_result, torch_result, op_name in test_cases:
        # zero grads
        val1.grad, val2.grad = (0, 0)
        ten1.grad, ten2.grad =  (torch.Tensor([0]), torch.Tensor([0]))

        # backward pass
        mcgrd_result.backward()
        torch_result.backward()

        msg = f'{op_name} backprop error'
        assert compare_results(val1.grad, ten1.grad.item()), msg
        assert compare_results(val2.grad, ten2.grad.item()), msg


test_binary_ops()
test_reverserd_binary_ops()
test_unary_ops()
test_backward()
print('Everything ok.')