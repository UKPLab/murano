from murano import LayerLocation, LogitLens, MuranoModel

model = MuranoModel.from_pretrained("gpt2")
lens = LogitLens()
locations = [LayerLocation(layers=slice(None))]

prompt = "The Eiffel Tower is in the city of"
artifact = model.run_with_lens(prompt, lens, locations)
lens.visualize(artifact)
