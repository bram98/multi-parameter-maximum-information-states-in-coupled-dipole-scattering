# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 15:31:37 2025

@author: 5914787
"""

results = {}
results["x"] = []
results["z"] = []
results["phi"] = []

sides = np.linspace(1, 3, num=50)

for side in sides:
    movable_dipoles = placedipoles.place_square_of_dipoles(side=side)
    frozen_dipoles = None

    dS_tensor = placedipoles.calculate_dS_for_3_parameters(
        sim_kwargs,
        angles_deg,
        movable_dipoles=movable_dipoles,
        frozen_dipoles=frozen_dipoles,
        delta_param=delta_param,
        flux_normalize=True,
    )

    fisher_calc = FisherMatrix(
        dS=dS_tensor, param_names=param_names, delta=delta_param
    )
    F_tensor_operator = fisher_calc.F

    base_sim_for_vis = CoupledDipoleSimulator(**_sim_params_vis)
    base_sim_for_vis.movable_scatterers = movable_dipoles
    base_sim_for_vis.frozen_scatterers = frozen_dipoles

    # Initialize Optimizer (used for Blocks 5, 6, and 7 onwards)
    optimizer_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for Optimizer: {optimizer_device}")
    optimizer = Optimizer(
        F_operator_tensor=F_tensor_operator, device=optimizer_device
    )

    loss_functions_dict = LossFunctionsDict()

    # Single parameters
    for i, param in enumerate(["x", "z", r"\phi"]):
        name = param.replace("\\", "")
        loss_functions_dict[name] = LossFunction(
            lossfunctions.single_param_F(i),
            name=name,
            displayname=f"Singe parameter ${param}$",
            # shortname=rf"$\mathcal{{J}}_{{{param}{param}}}$",
            shortname=rf"${param}$ (scalar)",
            maximize=True,
            method="eig",
        )

    loss_function: LossFunction
    for loss_function in loss_functions_dict.values():
        # if ("x_inv" not in loss_function.name) or "tr_inv" in loss_function.name:
        #     continue
        loss_function.optimize(
            optimizer,
            seed=2025,
            verbose=True,
            lr=0.1,
            lr_decay=0.8,
            lr_decay_step_size=500,
            steps=10000,
            multiruns=5,
        )
        results[loss_function.name].append(loss_function.bestval)

# %%

for name, value in results.items():
    plt.plot(sides, value, label=name)

plt.legend()
plt.show()
