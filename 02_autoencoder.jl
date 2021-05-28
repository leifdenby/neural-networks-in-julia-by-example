### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 8b41c1c0-a348-11eb-2938-f7d0c282449d
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

# ╔═╡ 0cc0ff7a-3cff-49de-b57d-f7b6cc29b45a
md"""
We're going to need some data to work with. So let's download a picture of a nice cloud (feel free to use your own example here!)
"""

# ╔═╡ 34847f89-4546-4f6d-ad21-20be7c9dfbd6
download("http://labs.sogeti.com/wp-content/uploads/2014/03/cloud.jpg", "cloud.jpg")

# ╔═╡ 6e2e9e25-25b1-4c06-9615-e1e63b629e16
cloud_img = load("cloud.jpg")

# ╔═╡ 8c26b9a1-e82d-49e7-b2fd-99ade1439c06
begin
	img_to_arr(img) = Float32.(permutedims(channelview(img), (2,3,1)))
	arr_to_img(arr) = colorview(RGB, permutedims(arr, (3,1,2)))
end

# ╔═╡ 0c2fa463-9252-43c7-9b05-10a5d7af4f9f
begin
	single_to_batch(x) = Flux.unsqueeze(x, 4)
	batch_to_single(x) = x[:,:,:,1]
end

# ╔═╡ 94b859ea-79c4-4b62-9c11-7df507f14ac6
function apply_model(img, model)
	arr = img_to_arr(img)
	x_batch = single_to_batch(arr)
	y_batch = model(x_batch)
	y_arr = batch_to_single(y_batch)
	arr_to_img(y_arr)
end

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
# ╟─8b41c1c0-a348-11eb-2938-f7d0c282449d
# ╟─0cc0ff7a-3cff-49de-b57d-f7b6cc29b45a
# ╠═34847f89-4546-4f6d-ad21-20be7c9dfbd6
# ╠═6e2e9e25-25b1-4c06-9615-e1e63b629e16
# ╠═8c26b9a1-e82d-49e7-b2fd-99ade1439c06
# ╠═0c2fa463-9252-43c7-9b05-10a5d7af4f9f
# ╠═94b859ea-79c4-4b62-9c11-7df507f14ac6
# ╠═9f123aab-eb82-433e-a511-338827432889
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
