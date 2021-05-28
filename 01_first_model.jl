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

# ╔═╡ bc1593d3-2eab-4306-9bb6-776fe2fbe82b
md"""
# Training our first model

Hi!

In this notebook we will find out how to create a model in `Flux` and how to train it. We'll be training a model to remove some random noise we've added to an image, so we will give it a noisy image and train it to produce the image we started with before adding noise.

To start out easy we'll be creating a really simple model which just looks at the neighbouring pixels and tries its best to use these to reconstruct the original image.

"""

# ╔═╡ 0cc0ff7a-3cff-49de-b57d-f7b6cc29b45a
md"""
We're going to need some data to work with. So let's download a picture of a nice cloud (feel free to use your own example here!)
"""

# ╔═╡ 34847f89-4546-4f6d-ad21-20be7c9dfbd6
download("http://labs.sogeti.com/wp-content/uploads/2014/03/cloud.jpg", "cloud.jpg")

# ╔═╡ 6e2e9e25-25b1-4c06-9615-e1e63b629e16
cloud_img = load("cloud.jpg")

# ╔═╡ 191cc3f8-d001-48d3-a094-b2c9dd967f5a
md"""
Let's have a look at what this image is actually made of once we've read in into julia. We can check the size of the image with `size` and using `typeof` we see that using `load` (from the `Images` package) we've gotten at 2D array of elements of the `RGB`-type (which are in itself stored as `UInt8`'s)."""

# ╔═╡ 2d8bf1c0-d06b-47ce-b84b-5f24a36c53f7
begin
	img = imresize(cloud_img, ratio=0.1)
	size(img), typeof(img)
end

# ╔═╡ 171202d4-7f42-4122-a434-787d86fb3b2b
md"""
But, when we're feeding data to the our model it will expect the RGB-values to be fed in as a 3rd dimension if the 3D array representing the image-extent and the three colour components (also called colour "channels" in `Flux`).

Fortunately the `Images` package comes with a very handy `channelview` method which we can use to get underlying 3D array data.
"""

# ╔═╡ 11330db6-0a7c-4e17-b832-c006acbd7d5c
arr = channelview(img)

# ╔═╡ 43d0fe0d-a729-48ad-a51b-9bd6c273599d
md"""
The last steps are 1) we need is to make the colour "channel" dimension be the last one (rather than the first) as `Flux` expects the input data to be shaped `HWCB`, *H*eight, Width, Channel and Batch-size (we'll get to batching below, but in essense this make it possible to pass in multiple images at once to our model, treating each of them identically). And 2) the models we will work with generally work best with floating point numbers (rather than integers).

To wrap all these transformations into little convenience functions we will define `image_to_arr` and `arr_to_img` below, the latter will be useful when we later want to look at the output of our model 🚀.
"""

# ╔═╡ 8c26b9a1-e82d-49e7-b2fd-99ade1439c06
begin
	img_to_arr(img) = Float32.(permutedims(channelview(img), (2,3,1)))
	arr_to_img(arr) = colorview(RGB, permutedims(arr, (3,1,2)))
end

# ╔═╡ a4d2fb7c-b1f3-4179-82c2-4f524ece4d9c
md"Let's check that our function works by making a heat-map plot of a single channel of our image turned into a 3D array of floats"

# ╔═╡ 6d2784d3-b8b8-41b6-bda6-68345ac3209d
heatmap(img_to_arr(img)[:,:,1])

# ╔═╡ e6c4a456-53c1-4686-aa65-7adfb6d94302
md"And using the `arr_to_img` function we get our original image back 😀"

# ╔═╡ 55a2c79f-7f64-469a-b074-6745cc2e8176
arr_to_img(img_to_arr(img))

# ╔═╡ bb1e4026-db05-4a74-934d-46dbc94797ff
md"""The very last thing we need to talk about before creating our first model is batching (or mini-batching). The idea here is that when we're training our model we don't want to improve on just a single training example at once (as that might lead to quite drastic changes to the model that only improve the results for that single example), but instead we want to make the model better for many different examples all at once (making changes that improve the model generally, rather than specifically for a single example).

To achieve this we can feed multiple training examples (images in our case) together to the model all at once, and this process is called *mini-batching*. This technique is so general that all neural network frameworks dedicate a specific dimension of the input (and output) data to this batching, the "batch dimension".

TODO: talk about noise here and adding multiple examples
"""

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

# ╔═╡ Cell order:
# ╟─bc1593d3-2eab-4306-9bb6-776fe2fbe82b
# ╟─8b41c1c0-a348-11eb-2938-f7d0c282449d
# ╟─0cc0ff7a-3cff-49de-b57d-f7b6cc29b45a
# ╠═34847f89-4546-4f6d-ad21-20be7c9dfbd6
# ╠═6e2e9e25-25b1-4c06-9615-e1e63b629e16
# ╟─191cc3f8-d001-48d3-a094-b2c9dd967f5a
# ╠═2d8bf1c0-d06b-47ce-b84b-5f24a36c53f7
# ╟─171202d4-7f42-4122-a434-787d86fb3b2b
# ╠═11330db6-0a7c-4e17-b832-c006acbd7d5c
# ╟─43d0fe0d-a729-48ad-a51b-9bd6c273599d
# ╠═8c26b9a1-e82d-49e7-b2fd-99ade1439c06
# ╟─a4d2fb7c-b1f3-4179-82c2-4f524ece4d9c
# ╠═6d2784d3-b8b8-41b6-bda6-68345ac3209d
# ╟─e6c4a456-53c1-4686-aa65-7adfb6d94302
# ╠═55a2c79f-7f64-469a-b074-6745cc2e8176
# ╟─bb1e4026-db05-4a74-934d-46dbc94797ff
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
