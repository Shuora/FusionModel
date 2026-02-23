import torch


def test_moe_distill_smoke():
    from src.fusion.distill import distillation_loss
    from src.fusion.moe_router import MoeRouter

    router = MoeRouter(input_dim=8, num_experts=3, num_classes=4)
    x = torch.randn(5, 8)
    out = router(x)

    assert out["logits"].shape == (5, 4)
    assert out["gates"].shape == (5, 3)

    teacher_logits = torch.randn(5, 4)
    student_logits = torch.randn(5, 4)
    labels = torch.tensor([0, 1, 2, 3, 1], dtype=torch.long)

    loss = distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.7)
    assert torch.is_tensor(loss)
    assert loss.ndim == 0
