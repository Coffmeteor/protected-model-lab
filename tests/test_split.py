import torch

from delta_split.coefficient import coefficient_factors
from delta_split.kron_svd import kron_singular_values, selected_direction_factors
from lora_mapping.lokr import dense_delta


def test_coefficient_reconstruction():
    torch.manual_seed(3)
    w1=torch.randn(3,2)
    w2=torch.randn(4,5)
    public,private=coefficient_factors(w1,w2,0.35)
    restored=dense_delta(*public)+dense_delta(*private)
    torch.testing.assert_close(restored,dense_delta(w1,w2),rtol=1e-6,atol=1e-6)


def test_lokr_strength_scaling_and_multiple_adapter_composition():
    torch.manual_seed(8)
    w1a,w2a=torch.randn(2,2),torch.randn(3,4)
    w1b,w2b=torch.randn(2,2),torch.randn(3,4)
    first=dense_delta(w1a,w2a,strength=0.75)
    second=dense_delta(w1b,w2b,strength=1.25)
    torch.testing.assert_close(first,torch.kron(w1a,w2a)*0.75)
    combined=first+second
    torch.testing.assert_close(combined,torch.kron(w1a,w2a)*0.75+torch.kron(w1b,w2b)*1.25)


def test_kron_singular_values_match_dense():
    torch.manual_seed(4)
    w1=torch.randn(3,2)
    w2=torch.randn(4,3)
    expected=torch.linalg.svdvals(torch.kron(w1,w2))
    actual=kron_singular_values(w1,w2)
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)


def test_all_selected_directions_reconstruct_kron():
    torch.manual_seed(5)
    w1=torch.randn(3,2)
    w2=torch.randn(4,3)
    up,down,_=selected_direction_factors(w1,w2,rank=6,strongest=True)
    torch.testing.assert_close(up@down,torch.kron(w1,w2),rtol=2e-5,atol=2e-5)
