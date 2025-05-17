import torch
from _network import NewCNNEncoder


def test_encoder_state_shape():
    mask_4x4 = torch.BoolTensor([
        [True, True, True, True, False],
        [True, True, True, True, False],
        [True, True, True, True, False],
        [True, True, True, True, False],
        [False, False, False, False, False],
    ])
    encoder = NewCNNEncoder(out_features=1000, mask=mask_4x4)  # out_features can be any valid value
    N = 2048  # number of games
    state = torch.randint(0, 16, (N, 25), dtype=torch.long)  # random board states
    print(f"State shape: {state.shape}, ndim: {state.ndim}")
    assert state.ndim == 2 and state.shape[1] == 25, f"Expected (N, 25), got {state.shape}"
    # Try passing through encoder
    out = encoder(state)
    print(f"Encoder output shape: {out.shape}")
    assert out.shape[0] == N, f"Encoder output batch size {out.shape[0]} != N ({N})"
    print("Encoder test passed.")


def main():
    test_encoder_state_shape()

if __name__ == "__main__":
    main()
