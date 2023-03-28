# OrthoAligner: Image-based Teeth Alignment Prediction via Latent Style Manipulation

把一个图片编辑的任务变成了一个潜空间探索的问题。

Our key innovation is that we formulate the image
translation problem from malpositioned teeth to aligned
teeth as a latent space exploration problem, where we first
model the teeth image manifold with the state-of-the-art
unsupervised GAN (i.e., StyleGAN in our paper) and find
the geometrically meaningful editing path that corresponds
to “alignment” in its latent space. To achieve so, we first
disentangle the teeth structure from other image features
and guide the editing using a ScireNet.

