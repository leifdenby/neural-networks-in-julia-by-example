### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ 3a537966-a37a-11eb-309d-83e7886daa79
begin
	using Flux
	using Plots
	using Images
	using ImageTransformations
end

# ╔═╡ 60c259b0-98dc-4c92-9a33-6f968214e985
download("http://labs.sogeti.com/wp-content/uploads/2014/03/cloud.jpg", "cloud.jpg")

# ╔═╡ 0b092962-70d5-4b3c-a23a-ed2c620c8d74
cloud_img = load("cloud.jpg")

# ╔═╡ fef35120-bf7f-4723-89df-f7326918057c
img = imresize(cloud_img, ratio=0.1)

# ╔═╡ 0504db57-2002-4801-b2bd-a5c7b4595305
begin
	img_to_arr(img) = Float32.(permutedims(channelview(img), (2,3,1)))
	arr_to_img(arr) = colorview(RGB, permutedims(arr, (3,1,2)))
end

# ╔═╡ e2a42226-50b3-465a-a626-17fd7c48dfa2
begin
	single_to_batch(x) = Flux.unsqueeze(x, 4)
	batch_to_single(x) = x[:,:,:,1]
	mask_to_arr(x) = Flux.unsqueeze(x, 3)
	arr_to_mask(x) = x[:,:,1]
end

# ╔═╡ d4dc6d71-3a68-448e-93a1-9840b66d3868
function apply_model(img, model)
	arr = img_to_arr(img)
	x_batch = single_to_batch(arr)
	y_batch = model(x_batch)
	y_arr = batch_to_single(y_batch)
	arr_to_img(y_arr)
end

# ╔═╡ d7e8df50-1189-46b7-b43c-282fb86629a4
begin
	function train_model(model, img, target, n_epochs::Int)
		function loss(x, y)
			Flux.Losses.mse(model(x), y)
		end
		
		x_batch = single_to_batch(img_to_arr(img))
		y_batch = single_to_batch(target)
		#y_batch = x_batch
		data = [(x_batch, y_batch)]
		
		losses = []
		function cb()
			append!(losses, loss(data[1]...))
		end
		
		params = Flux.params(model)
		
		opt = Flux.Optimise.Descent(0.1)
		Flux.@epochs n_epochs Flux.Optimise.train!(loss, params, data, opt; cb=Flux.throttle(cb, 1)
		)
		return losses
	end
	
	function train_and_plot_model(model, img, cloud_mask; n_epochs=20)
		target = cloud_mask |> mask_to_arr
		losses = train_model(model, img, target, n_epochs)
		target_predicted = model(img |> img_to_arr |> single_to_batch)
		cloud_mask_predicted = target_predicted |> batch_to_single |> arr_to_mask
		
		plot(layout=4)
		heatmap!(cloud_mask, subplot=1, title="target mask")
		heatmap!(cloud_mask_predicted, subplot=2, title="predicted mask")
		cmp_clipped = ifelse.(cloud_mask_predicted .> 0.5, 1.0, 0.0)
		heatmap!(cmp_clipped, subplot=3, title="predicted mask > 0.5")
		plot!(losses, subplot=4, title="losses during training")
	end
end

# ╔═╡ a9554a9e-e98d-4b5f-9ac2-e94fc1fe7bd0
md"""# U-Net

We now build a U-Net which is similar to an auto-encoder only it includes [skip-connections](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.SkipConnection) between the intermediate layers

Inspiration

- [https://github.com/rzietal/UNet.jl/blob/master/src/model.jl](https://github.com/rzietal/UNet.jl/blob/master/src/model.jl)
- [https://github.com/intelligenerator/unet](https://github.com/intelligenerator/unet)

![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fmiro.medium.com%2Fmax%2F1200%2F1*f7YOaE4TWubwaFF7Z1fzNw.png&f=1&nofb=1)
"""

# ╔═╡ e461e49d-fdda-4e24-9f26-c294f5b8591a
md"""
U-Nets are often used to predict one field from another which share spatial features.
For example we might want to predict a "cloud-mask" (i.e. where is there cloud and where isn't there?). This would be an example of image segmentation

Let's start by making a simple "cloud-mask" that we can use as our training target. We'll do this by converting the RGB colors into HSV [Hue Colour Saturation](https://juliaimages.org/latest/examples/color_channels/rgb_hsv_thresholding/#RGB-to-HSV-and-thresholding) and using putting a threshold on saturation to pick out
the bright cloud.
"""

# ╔═╡ b5c14c5f-a920-4ea8-9928-4d016669d6e0
begin
	img_saturation = channelview(HSV.(img))[2,:,:]
	cloud_mask = ifelse.(img_saturation .< 0.2, 1.0, 0.0)

	plot(layout=3)
	plot!(img, subplot=1)
	heatmap!(reverse(img_saturation, dims=1), subplot=2)
	heatmap!(reverse(cloud_mask, dims=1), subplot=3, colorbar=false)
end

# ╔═╡ 8b3e01be-5dab-48e5-8ee7-7704909caae7
md"""Builing a skip-connection"""

# ╔═╡ 25ffdbd7-1d22-4086-9a3b-5d146d3f3cff
begin
	# The size of the output of the down-scaling/up-scaling
	m_downscale_upscale = Chain(
		MaxPool((2,2)),
		Conv((3,3), 3=>6, pad=1),
		ConvTranspose((3, 3), 6=>6, stride=2),
		Conv((3,3), 6=>3, pad=1)
	)
	
	m_du_with_skip = Chain(
		# we concatenate along the channel dimension, doubling the number
		# channels
		SkipConnection(m_downscale_upscale, (mx, x) -> cat(mx, x, dims=3)), 
		# the number of channels is halved with a 1x1 convolution
		Conv((1,1), 6=>3)
	)
end

# ╔═╡ 96bb41f8-1f5d-4469-9bdb-d4fbc69030fc
md"""
Let's see what the different parts of this network does to a sample image.

We'll take a small subset of the image so the effect of max-pooling, up-scaling (with `ConvTranpose`) and creating a skip-connection (and then convolving across these extra channels).
"""

# ╔═╡ 340e7187-b254-41d0-be8f-908ac1942d59
begin
	img_small = img[100:150,100:150]
	img_mp = apply_model(img_small, MaxPool((2,2)))
	img_ct = apply_model(img_small, m_downscale_upscale)
	img_sk = apply_model(img_small, m_du_with_skip)
	
	plot(layout=4)
	plot!(img_small, subplot=1, title="input")
	plot!(img_mp, subplot=2, title="max-pool")
	plot!(img_ct, subplot=3, title="down-scaling\n+ up-scaling")
	plot!(img_sk, subplot=4, title="skip-connection\n+ convolution")
end

# ╔═╡ e20e8c33-e125-4db7-a3ec-8e763e7e7b8d
md"""Now let's make a network with the skip-connection architecture above, but with an added convolution as the end so we turn the three channels into the one for predicting out "cloud-mask"
"""

# ╔═╡ 85b1b8eb-2357-40da-acb3-bac560c68ee0
begin
	function one_layer_unet()
		# The size of the output of the down-scaling/up-scaling
		m_downscale_upscale = Chain(
			MaxPool((2,2)),
			Conv((3,3), 3=>6, pad=1),
			ConvTranspose((3, 3), 6=>6, stride=2),
			Conv((3,3), 6=>3, pad=1)
		)

		m_du_with_skip = Chain(
			# we concatenate along the channel dimension, doubling the number
			# channels
			SkipConnection(m_downscale_upscale, (mx, x) -> cat(mx, x, dims=3)), 
			# the number of channels is halved with a 1x1 convolution
			Conv((1,1), 6=>3),
			# finally we do a 1x1 convolution which combines the three remaining
			# channels into one (to predict the cloud-mask)
			Conv((1,1), 3=>1)
		)
	end
	m_unet1 = one_layer_unet()
	train_and_plot_model(m_unet1, img, Float32.(cloud_mask), n_epochs=20)
end

# ╔═╡ fc8a7656-745f-4fa9-acd0-c9d4146f5a57
md"""
We've now built our very first U-Net like architecture!

Next we will generalise this by making it deeper, producing coarser intermediate representations. Remember that for every down-scaling there will be a skip-connection
across to the up-scaling part of the network
"""

# ╔═╡ 2b6d6d6a-7035-45e8-a41b-f1baa3ccd6ba


# ╔═╡ Cell order:
# ╠═3a537966-a37a-11eb-309d-83e7886daa79
# ╠═60c259b0-98dc-4c92-9a33-6f968214e985
# ╠═0b092962-70d5-4b3c-a23a-ed2c620c8d74
# ╠═fef35120-bf7f-4723-89df-f7326918057c
# ╠═0504db57-2002-4801-b2bd-a5c7b4595305
# ╠═e2a42226-50b3-465a-a626-17fd7c48dfa2
# ╠═d4dc6d71-3a68-448e-93a1-9840b66d3868
# ╠═d7e8df50-1189-46b7-b43c-282fb86629a4
# ╟─a9554a9e-e98d-4b5f-9ac2-e94fc1fe7bd0
# ╟─e461e49d-fdda-4e24-9f26-c294f5b8591a
# ╠═b5c14c5f-a920-4ea8-9928-4d016669d6e0
# ╟─8b3e01be-5dab-48e5-8ee7-7704909caae7
# ╠═25ffdbd7-1d22-4086-9a3b-5d146d3f3cff
# ╟─96bb41f8-1f5d-4469-9bdb-d4fbc69030fc
# ╠═340e7187-b254-41d0-be8f-908ac1942d59
# ╟─e20e8c33-e125-4db7-a3ec-8e763e7e7b8d
# ╠═85b1b8eb-2357-40da-acb3-bac560c68ee0
# ╟─fc8a7656-745f-4fa9-acd0-c9d4146f5a57
# ╠═2b6d6d6a-7035-45e8-a41b-f1baa3ccd6ba
