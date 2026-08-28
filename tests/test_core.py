import pytest
import torch

from core_format.spcore import CoreLayer, ProtectedCore, read_core, write_core


def test_core_round_trip_and_carrier_binding(tmp_path):
    path=tmp_path/"x.spcore"
    core=ProtectedCore("mock","a"*64,"coefficient",(CoreLayer("x.weight",torch.eye(2),torch.ones(2,3),0.35),))
    write_core(path,core)
    loaded=read_core(path,"a"*64)
    assert loaded.architecture=="mock"
    torch.testing.assert_close(loaded.layers[0].w2,torch.ones(2,3))
    with pytest.raises(ValueError,match="carrier hash mismatch"):
        read_core(path,"b"*64)
