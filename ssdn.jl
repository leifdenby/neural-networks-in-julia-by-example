### A Pluto.jl notebook ###
# v0.14.3

using Markdown
using InteractiveUtils

# ╔═╡ ea1046a6-a470-11eb-29d3-3128f2297e71
begin
	using Flux
end

# ╔═╡ 57a0f39d-18d0-4d79-94e0-5f99e286d10b
md"""The aim of this notebook is to implemented the denoising network from the paper [High-Quality Self-Supervised Deep Image Denoising](https://arxiv.org/abs/1901.10277). As a guide we will be using the tensorflow implementation [on github](https://github.com/NVlabs/selfsupervised-denoising).
"""

# ╔═╡ 235a5cdd-00c6-4168-bbd7-b4d0b0c04e1c
md"""
We will start by implementing the "blind-spot convolution". From the paper:

> Convolution layers To restrict the receptive field of a zero-padding convolution layer to extend only, say, upwards, the easiest solution is to offset the feature maps downwards when performing the convolution operation. For an h × w kernel size, a downwards offset of k = ?h/2? pixels is equivalent to using a kernel that is shifted upwards so that all weights below the center row are zero. **Specifically, we first append k rows of zeros to the top of input tensor, then perform the convolution, and finally crop out the k bottom rows of the output**

Note that in the tensorflow implementation the tensors have shape `NCHW` (number in batch, channel number, height and width) whereas in `Flux` the shape is `WHCN`
"""

# ╔═╡ ac31e05f-524f-4aaf-b4f6-007583c3acb6
md"First we need a height-offset convolution"

# ╔═╡ 0256d08b-84b2-49d9-b78a-e994b11bd4c7
begin
	pad_offset(x, offset) = pad_zeros(x, (0,0, offset,0, 0,0, 0,0))
	
	conv_size = 3
	a = randn(Float32, (5, 5, 1, 1))
	offset = conv_size ÷ 2
	pad_offset(a, offset)
end

# ╔═╡ a0729868-a20e-4982-9f03-c30d59307e46
begin
	struct HeightOffsetConv2D
		filter::Tuple{<:Integer,<:Integer}
		ch::Pair{<:Integer,<:Integer}
	end
	
	function (c::HeightOffsetConv2D)(x::AbstractArray{T}) where T
		offset = c.filter[1] ÷ 2
		x_offset = pad_zeros(x, (0,0, offset,0, 0,0, 0,0))
		x_conv = Conv(c.filter, c.ch)(x_offset)
		return x_conv[:,1:size(x)[2]-1-offset,:,:]
	end
	
	# check that the shape is unchanged
	HeightOffsetConv2D((3,3),1=>1)(a) |> size == Conv((3,3), 1=>1)(a) |> size
end

# ╔═╡ b4ebf1c0-68a7-4372-bf9a-1e6c35b54255
md"Second we need to be able to rotate the tensors around the height/width dimensions"

# ╔═╡ fcb47b57-6126-4e05-be66-4ee2ae11b12c
begin
	function rotate_hw(x, angle)
		if angle == 0
			return x
		else
			x_rot = zeros(size(x))
			x_rot[:,:,1,1] = rotr90(x[:,:,1,1], angle ÷ 90)
			return x_rot
		end
	end
	
	rotate_hw(a, 90) == rotate_hw(a, -270)
end

# ╔═╡ d076aa88-ca21-440e-b514-a1dc22cf487b
begin
	x_input = randn(Float32, (10, 10, 1, 3))
	
	cat([rotate_hw(x_input, a) for a in [0, 90, 180, 270]]...; dims=4)
end

# ╔═╡ 8d229c01-2952-403c-9f93-9f9727090046
begin
	b = randn(Float32, (3, 3, 1))
	cat([b, b]...; dims=3)
end

# ╔═╡ Cell order:
# ╟─57a0f39d-18d0-4d79-94e0-5f99e286d10b
# ╠═ea1046a6-a470-11eb-29d3-3128f2297e71
# ╟─235a5cdd-00c6-4168-bbd7-b4d0b0c04e1c
# ╟─ac31e05f-524f-4aaf-b4f6-007583c3acb6
# ╠═0256d08b-84b2-49d9-b78a-e994b11bd4c7
# ╠═a0729868-a20e-4982-9f03-c30d59307e46
# ╟─b4ebf1c0-68a7-4372-bf9a-1e6c35b54255
# ╠═fcb47b57-6126-4e05-be66-4ee2ae11b12c
# ╠═d076aa88-ca21-440e-b514-a1dc22cf487b
# ╠═8d229c01-2952-403c-9f93-9f9727090046
