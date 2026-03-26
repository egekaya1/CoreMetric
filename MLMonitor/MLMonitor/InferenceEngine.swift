//
//  InferenceEngine.swift
//  CoreMetric
//
//  Created by Ege Kaya on 27.11.2025.
//

import Foundation
import CoreML
import Combine

class InferenceEngine: ObservableObject {
    @Published var currentScore: Double = 0.0
    @Published var alertThreshold: Double = 0.5
    @Published var isAnomalous: Bool = false
    @Published var modelStatus: String = "No model — tap Set Up to begin"
    @Published var isModelLoaded: Bool = false
    @Published var topSuspects: [SuspectProcess] = []

    private var baseThreshold: Double = 0.5
    private var model: MLModel?
    private var featureMeans: [Float] = []
    private var featureStds: [Float] = []

    /// Path where SetupOrchestrator saves the trained model.
    static var modelURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("CoreMetric/app/CoreMetric/Sources/Models/SystemMonitor.mlpackage")
    }

    init() {
        loadModel()
    }

    func loadModel() {
        let url = Self.modelURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            model = nil
            isModelLoaded = false
            modelStatus = "No model — tap Set Up to begin"
            return
        }

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            let compiled = try MLModel.compileModel(at: url)
            model = try MLModel(contentsOf: compiled, configuration: config)
            extractMetadata()
            isModelLoaded = true
            modelStatus = "Model Active"
        } catch {
            model = nil
            isModelLoaded = false
            modelStatus = "Error loading model: \(error.localizedDescription)"
        }
    }

    private func extractMetadata() {
        guard let model else { return }
        let meta = model.modelDescription.metadata
        guard let userDict = meta[.creatorDefinedKey] as? [String: String] else { return }

        if let meanStr = userDict["feature_means"], let stdStr = userDict["feature_stds"] {
            featureMeans = meanStr.split(separator: ",").compactMap { Float($0) }
            featureStds  = stdStr.split(separator: ",").compactMap { Float($0) }
        }

        if let threshStr = userDict["suggested_threshold"], let threshVal = Double(threshStr) {
            baseThreshold = threshVal
            let mult = UserDefaults.standard.double(forKey: "thresholdMultiplier")
            alertThreshold = threshVal * (mult > 0 ? mult : 1.0)
        }
    }

    /// Must be called on the main thread. Runs inference synchronously so the
    /// caller can immediately read updated published properties.
    func analyze(_ raw: [Float]) {
        guard let model, !featureMeans.isEmpty else {
            currentScore = 0.0
            isAnomalous  = false
            topSuspects  = []
            return
        }

        let inputSize = featureMeans.count

        do {
            let inputArray = try MLMultiArray(shape: [1, NSNumber(value: inputSize)], dataType: .float32)
            for i in 0..<inputSize where i < raw.count {
                let std  = (featureStds.count > i && featureStds[i] != 0) ? featureStds[i] : 1.0
                let mean = featureMeans.count > i ? featureMeans[i] : 0.0
                inputArray[i] = NSNumber(value: (raw[i] - mean) / std)
            }

            let provider = try MLDictionaryFeatureProvider(dictionary: ["input_features": inputArray])
            let output   = try model.prediction(from: provider)

            guard let reconstruction = output.featureValue(for: "reconstruction")?.multiArrayValue else { return }

            var mse: Float = 0.0
            for i in 0..<inputSize {
                let diff = inputArray[i].floatValue - reconstruction[i].floatValue
                mse += diff * diff
            }

            currentScore = Double(mse)
            let mult = UserDefaults.standard.double(forKey: "thresholdMultiplier")
            alertThreshold = baseThreshold * (mult > 0 ? mult : 1.0)
            isAnomalous    = currentScore > alertThreshold

            if isAnomalous {
                DispatchQueue.global(qos: .userInitiated).async {
                    let suspects = ProcessInspector.getTopProcesses()
                    DispatchQueue.main.async { self.topSuspects = suspects }
                }
            } else {
                topSuspects = []
            }
        } catch {
            // Inference errors are silent — avoid spamming logs every tick
        }
    }
}
