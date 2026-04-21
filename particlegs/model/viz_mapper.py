import torch
import torch.nn as nn


class VizMapper(nn.Module):
    """Unified VizMapper v4.

    Single MLP that takes (scale_factor, opacity_factor,
    gaussian_log_scale, gaussian_opacity_logit) and optionally (x, y, z) and
    outputs per-Gaussian (scale_correction, opacity_correction).

    v4 removes res_factor and distance inputs (proven unnecessary by C41b).
    All inputs share the full network capacity -- no base/adapter split.
    Zero-initialized last layer -> identity mapping at init.
    """

    def __init__(self, hidden_dim=64, num_layers=2, use_xyz=False,
                 factor_delta_scale=0.1, factor_delta_opacity=0.3,
                 min_opacity_clamp=0.4):
        super().__init__()

        input_dim = 7 if use_xyz else 4

        # Build network dynamically based on num_layers
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, 2))  # (delta_scale, delta_opacity)
        self.net = nn.Sequential(*layers)

        self.factor_delta_scale = factor_delta_scale
        self.factor_delta_opacity = factor_delta_opacity
        self.use_xyz = use_xyz
        self.min_opacity_clamp = min_opacity_clamp

        # Persist config for checkpoint serialization
        self.config = {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'use_xyz': use_xyz,
            'input_dim': input_dim,
            'min_opacity_clamp': min_opacity_clamp,
        }

        # Zero-init last layer -> identity mapping at start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, opacity, radius,
                gaussian_log_scale=None, gaussian_opacity_logit=None,
                xyz=None):
        """
        Args (all [N] per-Gaussian, per-camera values broadcast to N):
            opacity:  viz_opacity_factor
            radius:   viz_scale_factor
            gaussian_log_scale:     [N] per-Gaussian mean log scale (from _scaling)
            gaussian_opacity_logit: [N] per-Gaussian opacity logit (from _opacity)
            xyz:      [N, 3] per-Gaussian position (only used when use_xyz=True)
        Returns:
            scale_factor [N], opacity_factor [N]
        """
        N = radius.shape[0]
        device = radius.device

        # Center inputs so zero ~ typical operating point (matches zero-init identity prior)
        base_features = [
            radius - 1.0,
            opacity - 1.0,
            (gaussian_log_scale + 8.0) / 2.0 if gaussian_log_scale is not None else torch.zeros(N, device=device),
            (gaussian_opacity_logit + 3.0) / 4.0 if gaussian_opacity_logit is not None else torch.zeros(N, device=device),
        ]

        if self.use_xyz and xyz is not None:
            base_features.extend([xyz[:, 0], xyz[:, 1], xyz[:, 2]])

        inp = torch.stack(base_features, dim=-1)  # [N, 4] or [N, 7]

        out = self.net(inp)  # [N, 2]

        scale_corr = 1.0 + self.factor_delta_scale * torch.tanh(out[..., 0])
        opacity_corr = 1.0 + self.factor_delta_opacity * torch.tanh(out[..., 1])

        final_scale = radius * scale_corr
        final_opacity = torch.clamp(opacity * opacity_corr, min=self.min_opacity_clamp)

        return final_scale, final_opacity

    def load_from_old(self, old_state_dict):
        """Load old VizMapper (2-input, 4-input, or 6-input) into this net."""
        if any(k.startswith('base_net.') for k in old_state_dict):
            remap = {k.replace('base_net.', 'net.', 1): v
                     for k, v in old_state_dict.items() if k.startswith('base_net.')}
        else:
            remap = {k: v for k, v in old_state_dict.items() if k.startswith('net.')}

        cur = self.state_dict()
        loaded = 0
        for key, val in remap.items():
            if key in cur:
                if cur[key].shape == val.shape:
                    cur[key] = val
                    loaded += 1
                elif key == 'net.0.weight':
                    old_in = val.shape[1]
                    new_in = cur[key].shape[1]
                    min_out = min(cur[key].shape[0], val.shape[0])
                    common_in = min(old_in, new_in)
                    cur[key][:min_out, :common_in] = val[:min_out, :common_in]
                    if new_in > old_in:
                        cur[key][:, old_in:] = 0.0
                    loaded += 1
                elif key == 'net.0.bias':
                    min_out = min(cur[key].shape[0], val.shape[0])
                    cur[key][:min_out] = val[:min_out]
                    loaded += 1
        self.load_state_dict(cur)
        old_in_dim = remap.get('net.0.weight', torch.empty(0, 0)).shape[1] if 'net.0.weight' in remap else '?'
        print(f"Loaded {loaded} weights from old checkpoint ({old_in_dim}-input -> {self.config['input_dim']}-input)")
