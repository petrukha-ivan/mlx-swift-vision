//
//  ModelSelectionView.swift
//  MLX Vision
//
//  Created by Ivan Petrukha on 20.02.2026.
//

import SwiftUI
import MLXVision

enum ModelType: String, CaseIterable {
    case imageClassification = "Image Classification"
    case objectDetection = "Object Detection"
    case instanceSegmentation = "Instance Segmentation"
    case zeroShotClassification = "Zero-shot Classification"
    case zeroShotSegmentation = "Zero-shot Segmentation"
}

struct ModelSelection: Hashable {

    let id: String
    let revision: String
    let type: ModelType
    let quantizeBits: Int?
    let inputSize: Int?

    init(id: String, revision: String, type: ModelType, quantizeBits: Int? = nil, inputSize: Int? = nil) {
        self.id = id
        self.revision = revision
        self.type = type
        self.quantizeBits = quantizeBits
        self.inputSize = inputSize
    }
}

struct ModelSelectionView: View {

    @Binding var modelSelection: ModelSelection?

    var body: some View {
        Form {
            ForEach(ModelType.allCases, id: \.self) { type in
                Section(type.rawValue) {
                    ForEach(ModelSelection.presets.grouped[type, default: []], id: \.self) { preset in
                        Button {
                            modelSelection = preset
                        } label: {
                            HStack {

                                Text(preset.id)
                                    .lineLimit(1)
                                    .truncationMode(.tail)

                                Spacer()

                                if let quantizeBits = preset.quantizeBits {
                                    Text("\(quantizeBits)-bit")
                                        .padding(6)
                                        .font(.caption)
                                        .foregroundStyle(.white)
                                        .background(Color.accentColor, in: Capsule())
                                }

                                if let inputSize = preset.inputSize {
                                    Text("\(inputSize)x\(inputSize)")
                                        .padding(6)
                                        .font(.caption)
                                        .foregroundStyle(.white)
                                        .background(Color.accentColor, in: Capsule())
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// swift-format-ignore
extension ModelSelection {
    static let presets: [ModelSelection] = [
        ModelSelection(id: "microsoft/resnet-50", revision: "main", type: .imageClassification),
        ModelSelection(id: "google/efficientnet-b0", revision: "f86b650049699f1cf772edd370d7ea0382f0941f", type: .imageClassification),
        ModelSelection(id: "facebook/detr-resnet-50", revision: "main", type: .objectDetection),
        ModelSelection(id: "facebook/detr-resnet-50", revision: "main", type: .objectDetection, inputSize: 448),
        ModelSelection(id: "facebook/detr-resnet-101", revision: "main", type: .objectDetection),
        ModelSelection(id: "facebook/detr-resnet-101", revision: "main", type: .objectDetection, inputSize: 448),
        ModelSelection(id: "ipetrukha/rf-detr-nano", revision: "main", type: .objectDetection),
        ModelSelection(id: "ipetrukha/rf-detr-small", revision: "main", type: .objectDetection),
        ModelSelection(id: "PekingU/rtdetr_v2_r18vd", revision: "main", type: .objectDetection),
        ModelSelection(id: "PekingU/rtdetr_v2_r50vd", revision: "main", type: .objectDetection),
        ModelSelection(id: "AnnaZhang/lwdetr_tiny_60e_coco", revision: "main", type: .objectDetection),
        ModelSelection(id: "AnnaZhang/lwdetr_small_60e_coco", revision: "main", type: .objectDetection),
        ModelSelection(id: "facebook/detr-resnet-50-panoptic", revision: "12df956224e66b0faed42e288f43704ddab668ce", type: .instanceSegmentation),
        ModelSelection(id: "facebook/detr-resnet-50-panoptic", revision: "12df956224e66b0faed42e288f43704ddab668ce", type: .instanceSegmentation, inputSize: 448),
        ModelSelection(id: "facebook/detr-resnet-101-panoptic", revision: "main", type: .instanceSegmentation),
        ModelSelection(id: "facebook/detr-resnet-101-panoptic", revision: "main", type: .instanceSegmentation, inputSize: 448),
        ModelSelection(id: "openai/clip-vit-base-patch16", revision: "5ef227a78de3f75873f373246dac80def63b0003", type: .zeroShotClassification),
        ModelSelection(id: "openai/clip-vit-base-patch32", revision: "c237dc49a33fc61debc9276459120b7eac67e7ef", type: .zeroShotClassification),
        ModelSelection(id: "facebook/sam3", revision: "main", type: .zeroShotSegmentation),
        ModelSelection(id: "facebook/sam3", revision: "main", type: .zeroShotSegmentation, inputSize: 672),
        ModelSelection(id: "facebook/sam3", revision: "main", type: .zeroShotSegmentation, inputSize: 336),
        ModelSelection(id: "facebook/sam3", revision: "main", type: .zeroShotSegmentation, quantizeBits: 4, inputSize: 672),
        ModelSelection(id: "facebook/sam3", revision: "main", type: .zeroShotSegmentation, quantizeBits: 4, inputSize: 336),
    ]
}

extension Array<ModelSelection> {
    var grouped: Dictionary<ModelType, [ModelSelection]> {
        Dictionary(grouping: self, by: \.type)
    }
}
