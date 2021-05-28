### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ ea1046a6-a470-11eb-29d3-3128f2297e71
# to allow this notebook to run on github-pages we have to create an environment
# on-the-fly into which we install the packages we need. If you're running this
# notebook locally on your computer you can just remove the "Pkg"-related commands
# below (and then use the environment from which you started pluto")
begin
	import Pkg
    # activate a clean environment
    Pkg.activate(mktempdir())

    Pkg.add([
        Pkg.PackageSpec(name="Flux"),
        Pkg.PackageSpec(name="Plots"),
		Pkg.PackageSpec(name="Images"),
		Pkg.PackageSpec(name="ImageTransformations"),
    ])
	
	using Flux
	using Plots
	using Images
	using ImageTransformations
end

# ╔═╡ 57a0f39d-18d0-4d79-94e0-5f99e286d10b
md"""
# Self-Supervised Image denoising

The aim of this notebook is to implement the denoising network from the paper [High-Quality Self-Supervised Deep Image Denoising](https://arxiv.org/abs/1901.10277). As a guide we will be using the tensorflow implementation [on github](https://github.com/NVlabs/selfsupervised-denoising).
"""

# ╔═╡ 235a5cdd-00c6-4168-bbd7-b4d0b0c04e1c
md"""
We will start by implementing the "blind-spot convolution". From the paper:

> Convolution layers: To restrict the receptive field of a zero-padding convolution layer to extend only, say, upwards, the easiest solution is to offset the feature maps downwards when performing the convolution operation. For an h × w kernel size, a downwards offset of k = ?h/2? pixels is equivalent to using a kernel that is shifted upwards so that all weights below the center row are zero. **Specifically, we first append k rows of zeros to the top of input tensor, then perform the convolution, and finally crop out the k bottom rows of the output**

Note that in the tensorflow implementation the tensors have shape `NCHW` (number in batch, channel number, height and width) whereas in `Flux` the shape is `WHCN`
"""

# ╔═╡ ac31e05f-524f-4aaf-b4f6-007583c3acb6
md"First we need a height-offset convolution"

# ╔═╡ 0256d08b-84b2-49d9-b78a-e994b11bd4c7
begin
	pad_offset(x, offset) = pad_zeros(x, (0,0, offset,0, 0,0, 0,0))
	
	N = 256
	conv_size = 3
	a = randn(Float32, (N, N, 1, 1))
	offset = conv_size ÷ 2
	pad_offset(a, offset)
end

# ╔═╡ a0729868-a20e-4982-9f03-c30d59307e46
begin
	struct HeightOffsetConv2D
		filter::Tuple{<:Integer,<:Integer}
		ch::Pair{<:Integer,<:Integer}
		activation::Function
	end
	
	function (c::HeightOffsetConv2D)(x::AbstractArray{T}) where T
		offset = c.filter[1] ÷ 2
		x_offset = pad_zeros(x, (0,0, offset,0, 0,0, 0,0))
		x_conv = Conv(c.filter, c.ch, c.activation, pad=1)(x_offset)
		return x_conv[:,1:size(x_conv)[2]-offset,:,:]
	end
	
	# check that the shape is unchanged
	HeightOffsetConv2D((3,3),1=>1, identity)(a) |> size, Conv((3,3), 1=>1)(a) |> size
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
	struct HeightOffsetMaxPool
		filter::Tuple{<:Integer,<:Integer}
	end
	
	function (c::HeightOffsetMaxPool)(x::AbstractArray{T}) where T
		offset = c.filter[1] ÷ 2
		x_offset = pad_zeros(x, (0,0, offset,0, 0,0, 0,0))[:,1:size(x)[2],:,:]
		return MaxPool(c.filter)(x_offset)
	end
	
	# check that the shape is unchanged
	HeightOffsetMaxPool((3,3))(a) |> size, MaxPool((3,3))(a) |> size
end

# ╔═╡ 8d229c01-2952-403c-9f93-9f9727090046
begin
	layers = Vector{Any}()
	
	# create four rotated copies and stack along batch dimension
	rotated_stack(x) = cat([rotate_hw(x, a) for a in [0, 90, 180, 270]]...; dims=4)
	
	append!(layers, [rotated_stack])
	
	nc_in = 3
	nc_hd = 4
	w = 3
	
	LR = leakyrelu
	
	append!(layers, [
			HeightOffsetConv2D((w, w), nc_in=>nc_hd, LR),
			HeightOffsetConv2D((w, w), nc_hd=>nc_hd, LR),
			HeightOffsetMaxPool((2,2)),
	])
	
	image = randn(Float32, (256, 256, 3, 1))
	Chain(layers...)(image) |> size
end

# ╔═╡ 623a0363-b324-4d03-bd80-1e0ed7c69300
begin
	x_img = randn(Float32, (4, 4, 1, 1))
	#Upsample((2,2))(x_img) |> size, HeightOffsetMaxPool((2,2))(x_img) |> size
	ConvTranspose((2,2), 1=>2)(x_img)
end

# ╔═╡ 99fb471d-60d1-4bc9-9f72-92e9c9bdc3b3
begin
	level1 = Chain(
		# offset convolution with padding
		HeightOffsetConv2D((w,w), nc_hd => nc_hd, LR),
		# max-pool coarsening
		HeightOffsetMaxPool((2,2)),
		# offset convolution with padding
		HeightOffsetConv2D((w,w), nc_hd => nc_hd, LR),
		# conv-tranpose up, doubling number of channels
		ConvTranspose((2,2), nc_hd => 2*nc_hd, stride=2)
	)
		
	level2 = Chain(
		# offset convolution with padding
		HeightOffsetConv2D((w,w), nc_hd => nc_hd, LR),
		# max-pool coarsening
		HeightOffsetMaxPool((2,2)),
		# skip-connetion with concatenation across lower level
		SkipConnection(level1, (mx, x) -> cat(mx, x, dims=3)),
		# conv-transpose up in lower level doubled number channels
		# so with concat that is (2+1)*nc_hd. Conv with padding
		# to return to nc_hd channels
		HeightOffsetConv2D((w,w), nc_hd*3 => nc_hd, LR),
		# conv with padding to return to nc_hd channels
		HeightOffsetConv2D((w,w), nc_hd => nc_hd, LR),
		# conv-transpose up doubling number of channels
		ConvTranspose((2,2), nc_hd => 2*nc_hd, stride=2)
	)
	
		
	level3 = Chain(
		# offset convolution with padding
		HeightOffsetConv2D((w,w), nc_hd => nc_hd, LR),
		# max-pool coarsening
		HeightOffsetMaxPool((2,2)),
		# skip-connetion with concatenation across lower level
		SkipConnection(level2, (mx, x) -> cat(mx, x, dims=3)),
		# conv-transpose up in lower level doubled number channels
		# so with concat that is (2+1)*nc_hd. Conv with padding
		# to return to nc_hd channels
		HeightOffsetConv2D((w,w), nc_hd*3 => nc_hd, LR),
		# conv with padding to return to nc_hd channels
		HeightOffsetConv2D((w,w), nc_hd => nc_hd, LR),
		# conv-transpose up doubling number of channels
		ConvTranspose((2,2), nc_hd => 2*nc_hd, stride=2)
	)
	
	num_features = 8
	model = Chain(
		level1,
		HeightOffsetConv2D((w,w), 2*nc_hd => num_features, LR)
	)
	img_intermediate = randn(Float32, (256, 256, nc_hd, 1))
	model(img_intermediate) |> size
end

# ╔═╡ cbab1f07-3aa5-4c79-ad5b-ae69b35db8eb
begin
	function make_ssdn(nc_in, n_levels, n_out_features)
		lower_level = undef
		
		for n in 1:n_levels
			layers = [
				# offset convolution with padding
				HeightOffsetConv2D((w,w), nc_hd => nc_hd, LR),
				# max-pool coarsening
				HeightOffsetMaxPool((2,2))
			]
			
			if n > 1
				append!(layers, [
					# skip-connetion with concatenation across lower level
					SkipConnection(lower_level, (mx, x) -> cat(mx, x, dims=3))
					# conv-transpose up in lower level doubled number channels
					# so with concat that is (2+1)*nc_hd. Conv with padding
					# to return to nc_hd channels
					HeightOffsetConv2D((w,w), nc_hd*3 => nc_hd, LR)
				])
			end

			append!(layers,
				[
				# offset convolution with padding
				HeightOffsetConv2D((w,w), nc_hd => nc_hd, LR),
				# conv-tranpose up, doubling number of channels
				ConvTranspose((2,2), nc_hd => 2*nc_hd, stride=2)
			])
			
			lower_level = Chain(layers...)
		end
		
		# TODO: add in image stacking and unstacking etc
		model = Chain(
			HeightOffsetConv2D((w,w), nc_in => nc_hd, LR),
			lower_level,
			HeightOffsetConv2D((w,w), 2*nc_hd => n_out_features, LR)
		)
		return model
	end
	
	model_ssdn = make_ssdn(3, 5, 7)
	model_ssdn(image) |> size, image |> size
end

# ╔═╡ 53db6c56-8e3d-40af-b0e7-1df9ea555b88
model_ssdn

# ╔═╡ Cell order:
# ╟─57a0f39d-18d0-4d79-94e0-5f99e286d10b
# ╟─ea1046a6-a470-11eb-29d3-3128f2297e71
# ╟─235a5cdd-00c6-4168-bbd7-b4d0b0c04e1c
# ╟─ac31e05f-524f-4aaf-b4f6-007583c3acb6
# ╠═0256d08b-84b2-49d9-b78a-e994b11bd4c7
# ╠═a0729868-a20e-4982-9f03-c30d59307e46
# ╟─b4ebf1c0-68a7-4372-bf9a-1e6c35b54255
# ╠═fcb47b57-6126-4e05-be66-4ee2ae11b12c
# ╠═d076aa88-ca21-440e-b514-a1dc22cf487b
# ╠═8d229c01-2952-403c-9f93-9f9727090046
# ╠═623a0363-b324-4d03-bd80-1e0ed7c69300
# ╠═99fb471d-60d1-4bc9-9f72-92e9c9bdc3b3
# ╠═cbab1f07-3aa5-4c79-ad5b-ae69b35db8eb
# ╠═53db6c56-8e3d-40af-b0e7-1df9ea555b88
