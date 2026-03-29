# MLX Vision

[MLX](https://github.com/ml-explore/mlx-swift) Vision is a Swift library for running computer vision models on Apple Silicon. The library provides a flexible API for various tasks: image classification, object detection, image segmentation, zero-shot classification, zero-shot segmentation, and embedding extraction.

## Supported Tasks and Models

| Task | Models |
| - | - |
| Image Classification | ResNet, EfficientNet |
| Object Detection | DETR, RF-DETR, LW-DETR, RT-DETR |
| Instance Segmentation | DETR, RF-DETR |
| Zero-shot Classification | CLIP, SigLIP |
| Zero-shot Segmentation | SAM 3 |
| Embeddings Extraction | CLIP, SigLIP |

## Installation

Add the following to your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/petrukha-ivan/mlx-swift-vision", from: "0.0.1")
]
```

Then add the library as a dependency for your targets:

```swift
dependencies: [
    .product(name: "MLXVision", package: "mlx-swift-vision")
]
```

## Usage

### Load a Model

Model loading is unified. Pass a model id and task type. The factory resolves the model architecture and returns a typed pipeline with input and output types bound to the task.

```swift
import MLXVision

let factory = ModelFactory.shared
let model = try await factory.load("microsoft/resnet-50", for: ImageClassificationTask.self)
```

### Model Compression

You can override the default model configuration. For example, with the following configuration, it is possible to run the SAM 3 model even on an iPhone. By default, all models are loaded in `bfloat16` format.

```swift
let inputSize = CGSize(width: 336.0, height: 336.0)
let overrides = ModelOverrides(inputSize: inputSize, quantizeBits: 4)

let factory = ModelFactory.shared
let model = try await factory.load("facebook/sam3", for: ZeroShotSegmentationTask.self, overrides: overrides)
```

### Run a Model

### Image Classification

```swift
let request = ImageClassificationRequest(image: image)
let results = try model(request).top(5) // Top 5 items sorted by score
let summary = results.map { "\($0.label) - \($0.score)" } // Labels with scores
```

### Object Detection

```swift
let request = ObjectDetectionRequest(image: image)
let result = try model(request).top(1)[0] // Top detection result
let bbox = result.bbox // Normalized CGRect
```

### Instance Segmentation

```swift
let request = InstanceSegmentationRequest(image: image)
let result = try model(request).top(1)[0] // Top segmentation result
let mask = result.mask // Grayscale CIImage mask
```

### Zero-shot Classification

```swift
let labels = ["cat", "dog", "car"]
let request = ZeroShotClassificationRequest(image: image, labels: labels)
let results = try model(request) // 3 items with scores for each provided label
```

### Zero-shot Segmentation

```swift
let prompt = "orange cat"
let request = ZeroShotSegmentationRequest(image: image, prompt: prompt)
let results = try model(request) // N items with masks matching prompt description
```

## Image Annotation

Models return raw results: classification scores, normalized bounding boxes, and segmentation masks. You can process these results manually, or use image annotators:

```swift
let request = ObjectDetectionRequest(image: image)
let results = try model(request)

let annotator = BoxAnnotator(lineWidth: 8.0) // See also LabelAnnotator, MaskAnnotator
let annotatedImage = annotator.annotate(image: image, detections: results)
```

## Examples

This repository includes a fully featured iOS/macOS app. You can find more usage examples inside. It includes photo library processing and live-camera processing to test real-time model performance. Build the project in Release configuration to ensure the best performance.

<div style="display:flex;">
  <img width="49%" alt="556168490-fec2dec1-207b-4e2d-ae96-21eccc5d7aac" src="https://github.com/user-attachments/assets/7e9960b1-1d97-43ef-8c44-a734565b1513" />
  <img width="49%" alt="556168534-efe448bd-ca48-45b3-8424-07ca5aace941" src="https://github.com/user-attachments/assets/7071ad63-8784-43e8-9877-5503127e8b29" />
</div>

<div style="display:flex;">
  <img width="49%" alt="556161891-16a0e144-1127-40dc-b10e-4c7ea1bd5d2c" src="https://github.com/user-attachments/assets/fc106032-18bc-4605-ba55-00aff08c7250" />
  <img width="49%" alt="556162008-d23803b5-3a33-4df6-959d-b2a376480587" src="https://github.com/user-attachments/assets/939b6302-317c-4e8d-9266-b4879732fed9" />
</div>

<div style="display:flex;">
  <img width="49%" alt="556162097-5b270022-54ff-485b-b1cb-2ec62117fbc0" src="https://github.com/user-attachments/assets/a7aea0b5-f2e9-46ad-89b9-e84a557944cd" />
  <img width="49%" alt="556162170-d9e983ee-93ed-4f32-90de-82be07abd07c" src="https://github.com/user-attachments/assets/f0da5cc4-acba-4d46-82b9-ac860df984f2" />
</div>

## Performance

This is not a comprehensive and accurate test, the numbers below are approximate live-camera inference measurements. Models tested with `bfloat16` dtype without any quantization.

Metrics on M3 Max:

| Model | Input Size | Average Processing Time | Frames per Second |
| - | - | - | - |
| roboflow/rf-detr-nano | 384x384 | 10 ms | 102 |
| roboflow/rf-detr-seg-nano | 384x384 | 21 ms | 48 |
| facebook/detr-resnet-50 | 448x448 | 12 ms | 82 |
| facebook/detr-resnet-50-panoptic | 448x448 | 48 ms | 20 |
| facebook/sam3 | 336x336 | 120 ms | 8 |

Metrics on iPhone 16 Pro Max are obviously lower, but still practical even for interactive use:

| Model | Input Size | Average Processing Time | Frames per Second |
| - | - | - | - |
| roboflow/rf-detr-nano | 384x384 | 32 ms | 31 |
| roboflow/rf-detr-seg-nano | 384x384 | 85 ms | 11 |
| facebook/detr-resnet-50 | 448x448 | 40 ms | 25 |
| facebook/detr-resnet-50-panoptic | 448x448 | 230 ms | 4 |
| facebook/sam3 | 336x336 | 500 ms | 2 |

## Legal Notes

This library does not redistribute model resources such as weights or tokenizers. You must obtain an access token if model access is limited. While the library has a permissive license, you still have to comply with each model-specific license.

## Troubleshooting

This library is in an early stage of development. If you encounter a problem, please create an issue or open a pull request. Contributions are welcome!
