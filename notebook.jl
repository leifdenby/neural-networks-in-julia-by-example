### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ 8b41c1c0-a348-11eb-2938-f7d0c282449d
begin
	using Flux
	using Plots
	using Images
	using ImageTransformations
end

# ╔═╡ 34847f89-4546-4f6d-ad21-20be7c9dfbd6
download("http://labs.sogeti.com/wp-content/uploads/2014/03/cloud.jpg", "cloud.jpg")

# ╔═╡ 6e2e9e25-25b1-4c06-9615-e1e63b629e16
cloud_img = load("cloud.jpg")

# ╔═╡ 2d8bf1c0-d06b-47ce-b84b-5f24a36c53f7
begin
	img = imresize(cloud_img, ratio=0.1)
	size(img), typeof(img)
end

# ╔═╡ 11330db6-0a7c-4e17-b832-c006acbd7d5c
arr = channelview(img)

# ╔═╡ 8c26b9a1-e82d-49e7-b2fd-99ade1439c06
begin
	img_to_arr(img) = Float32.(permutedims(channelview(img), (2,3,1)))
	arr_to_img(arr) = colorview(RGB, permutedims(arr, (3,1,2)))
end

# ╔═╡ 6d2784d3-b8b8-41b6-bda6-68345ac3209d
heatmap(img_to_arr(img)[:,:,1])

# ╔═╡ 55a2c79f-7f64-469a-b074-6745cc2e8176
arr_to_img(img_to_arr(img))

# ╔═╡ 0c2fa463-9252-43c7-9b05-10a5d7af4f9f
begin
	single_to_batch(x) = Flux.unsqueeze(x, 4)
	batch_to_single(x) = x[:,:,:,1]
end

# ╔═╡ 60827b5b-0e12-44bc-a791-51228d436650
md"How does a convolution work?"

# ╔═╡ fc53568e-46d8-46b9-a3f4-822fa6134b27
Flux.outdims(Conv((3, 3), 3 => 3), (10, 10, 3, 1))

# ╔═╡ 663fba9b-eb1b-4cd6-9790-8bb7103c91b2
md"""How to transpose convolutions work?

![](http://ashukumar27.io/assets/neuralnets/transpose_conv.jpg)
"""

# ╔═╡ 45478e4b-f20c-4e61-b651-b81eca36b034
Flux.outdims(ConvTranspose((3, 3), 3 => 3), (10, 10, 3, 1))

# ╔═╡ 92dfd85e-3bca-4002-96ee-30a158f82156
md"# Applying model to a single image"

# ╔═╡ 9e44ae15-8470-4962-b4ea-460437baa658
model_simple = Chain(Conv((3,3), 3 => 3, identity, pad=1))

# ╔═╡ 94b859ea-79c4-4b62-9c11-7df507f14ac6
function apply_model(img, model)
	arr = img_to_arr(img)
	x_batch = single_to_batch(arr)
	y_batch = model(x_batch)
	y_arr = batch_to_single(y_batch)
	arr_to_img(y_arr)
end

# ╔═╡ c78a466d-b64d-4799-94bc-cea0d984bf41
apply_model(img, model_simple)

# ╔═╡ 9f123aab-eb82-433e-a511-338827432889
begin
	function train_model(model, img)
		function loss(x, y)
			Flux.Losses.mse(model(x), y)
		end
		
		x_batch = single_to_batch(img_to_arr(img))
		data = [(x_batch, x_batch)]
		
		losses = []
		function cb()
			append!(losses, loss(x_batch, x_batch))
		end
		
		params = Flux.params(model)
		
		opt = Flux.Optimise.Descent(0.1)
		Flux.@epochs 20 Flux.Optimise.train!(loss, params, data, opt; cb=Flux.throttle(cb, 1)
		)
		return losses
	end
	
	function train_and_plot_model(model, img)
		losses = train_model(model_simple, img)
		img_predicted = apply_model(img, model_simple)
		plot(layout=2)
		plot!(img_predicted, subplot=1, title="predicted image")
		plot!(losses, subplot=2, title="losses during training")
	end
end

# ╔═╡ 7ad87cf2-90e3-4bef-93a6-2aa498a2cd62
train_and_plot_model(model_simple, img)

# ╔═╡ 5b5a66f7-7370-4376-a88f-a6dcf7ffff50
md"""# Auto-encoder

Now we move to creating a more complicated model architecture: an auto-encoder
"""

# ╔═╡ c09fd1fe-00aa-47d2-8737-27c5351028b1
md"""## Generic Auto-encoder"""

# ╔═╡ 52d2161a-e1c2-4264-a4a0-5e990928a43b
begin
	struct AutoEncoder
		encoder
		decoder
	end
	
	Flux.@functor AutoEncoder
	
	function build_encoder(num_layers::Int, num_channels::Int; activation=identity, use_batchnorm=false)
		layers = []
		for i in 1:num_layers
			nc_in = num_channels*i
			nc_out = num_channels*(i+1)
			if !use_batchnorm
				append!(layers, [Conv((3, 3), nc_in => nc_out, activation)])
			else
				append!(layers, [Conv((3, 3), nc_in => nc_out), BatchNorm(nc_out, activation)])
			end				
		end
		return Chain(layers...)
	end
	
	function build_decoder(num_layers::Int, num_channels::Int; activation=identity)
		layers = []
		for i in 1:num_layers
			nc_in = num_channels*(num_layers-i+2)
			nc_out = num_channels*(num_layers-i+1)
			append!(layers, [ConvTranspose((3, 3), nc_in => nc_out), x-> activation.(x)])
		end
		return Chain(layers...)
	end
	
	function AutoEncoder(num_layers :: Int, num_channels :: Int; activation=identity, use_batchnorm=true)
		encoder = build_encoder(num_layers, num_channels; activation=activation, use_batchnorm=use_batchnorm)
		decoder = build_decoder(num_layers, num_channels; activation=activation)
		return AutoEncoder(encoder, decoder)
	end

	function (ae::AutoEncoder)(x::AbstractArray{T}) where T
		return ae.decoder(ae.encoder(x))
	end
	
	function forward(ae::AutoEncoder, x::AbstractArray{T}) where T
		ae.encoder(x)
	end
end

# ╔═╡ d3d143c1-fd96-4a9c-91a1-e56f512b8abc
begin
	struct AutoEncoderSimple
		encoder
		decoder
	end
	
	Flux.@functor AutoEncoderSimple
	
	function AutoEncoderSimple(num_channels::Int)
		encoder = Conv((3,3), num_channels => num_channels*2)
		decoder = ConvTranspose((3,3), num_channels*2 => num_channels)
		return AutoEncoder(encoder, decoder)
	end

	function (ae::AutoEncoderSimple)(x::AbstractArray{T}) where T
		return ae.decoder(ae.encoder(x))
	end
end

# ╔═╡ 9612843b-7003-44fe-8473-2706f323d14c
model_ae_simple = AutoEncoderSimple(3)

# ╔═╡ b925b4d3-822d-4a97-bc96-ac9c6fb20266
apply_model(img, model_ae_simple)

# ╔═╡ b35ba320-8401-4090-9195-d07231670d75
train_and_plot_model(model_ae_simple, img)

# ╔═╡ cf6e4b1f-afc4-4f40-9617-46135a788151
model_ae = AutoEncoder(2,3; activation=relu)

# ╔═╡ 46b7699a-1ada-483a-b2d3-c27a37425cb4
apply_model(img, model_ae)

# ╔═╡ 9d4f5119-5569-4892-b509-c50be5a3b74c
train_and_plot_model(model_ae, img)

# ╔═╡ Cell order:
# ╠═8b41c1c0-a348-11eb-2938-f7d0c282449d
# ╠═34847f89-4546-4f6d-ad21-20be7c9dfbd6
# ╠═6e2e9e25-25b1-4c06-9615-e1e63b629e16
# ╠═2d8bf1c0-d06b-47ce-b84b-5f24a36c53f7
# ╠═11330db6-0a7c-4e17-b832-c006acbd7d5c
# ╠═8c26b9a1-e82d-49e7-b2fd-99ade1439c06
# ╠═6d2784d3-b8b8-41b6-bda6-68345ac3209d
# ╠═55a2c79f-7f64-469a-b074-6745cc2e8176
# ╠═0c2fa463-9252-43c7-9b05-10a5d7af4f9f
# ╠═60827b5b-0e12-44bc-a791-51228d436650
# ╠═fc53568e-46d8-46b9-a3f4-822fa6134b27
# ╟─663fba9b-eb1b-4cd6-9790-8bb7103c91b2
# ╠═45478e4b-f20c-4e61-b651-b81eca36b034
# ╟─92dfd85e-3bca-4002-96ee-30a158f82156
# ╠═9e44ae15-8470-4962-b4ea-460437baa658
# ╠═94b859ea-79c4-4b62-9c11-7df507f14ac6
# ╠═c78a466d-b64d-4799-94bc-cea0d984bf41
# ╠═9f123aab-eb82-433e-a511-338827432889
# ╠═7ad87cf2-90e3-4bef-93a6-2aa498a2cd62
# ╟─5b5a66f7-7370-4376-a88f-a6dcf7ffff50
# ╠═d3d143c1-fd96-4a9c-91a1-e56f512b8abc
# ╠═9612843b-7003-44fe-8473-2706f323d14c
# ╠═b925b4d3-822d-4a97-bc96-ac9c6fb20266
# ╠═b35ba320-8401-4090-9195-d07231670d75
# ╟─c09fd1fe-00aa-47d2-8737-27c5351028b1
# ╠═52d2161a-e1c2-4264-a4a0-5e990928a43b
# ╠═cf6e4b1f-afc4-4f40-9617-46135a788151
# ╠═46b7699a-1ada-483a-b2d3-c27a37425cb4
# ╠═9d4f5119-5569-4892-b509-c50be5a3b74c
