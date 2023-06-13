# Schrodinger-equation-DNN

In this project, I am going to display the power of Deep Neural Networks (Multi-layered perceptrons) to solve the 1D time-independent Schrödinger equation. The network will essentially learn a non-linear map between the potential of a particle and the associated wave function of a particle.

If a particle is subject to a known potential $$\mathbb{V}(\boldsymbol{x})$$, we need to find the find it's wave function $$\boldsymbol{\psi}$$ that satisfies the following relation:
$$-\frac{\boldsymbol{\Sigma}_m(\boldsymbol{x})}{2}\boldsymbol{\Delta\psi}(\boldsymbol{x})+\mathbb{V}(\boldsymbol{x})\boldsymbol{\psi}(\boldsymbol{x})=\mathbb{E}\boldsymbol{\psi}(\boldsymbol{x})$$
