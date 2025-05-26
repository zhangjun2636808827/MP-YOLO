from ultralytics import YOLO, checks, hub
checks()

hub.login('b3ad0fbd3d3464e26433f4fc7d4f59bcfafd389a2a')

model = YOLO('https://hub.ultralytics.com/models/1My1YOgSSHE6K1owtS7E')
results = model.train()