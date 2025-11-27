//
//  InferenceEngine.swift
//  MLMonitor
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
    @Published var modelStatus: String = "Initializing..."
    @Published var topSuspects: [SuspectProcess] = []
    
    // We hold the generic base class here
    private var model: MLModel?
    private var featureMeans: [Float] = []
    private var featureStds: [Float] = []
    
    init() {
        loadModel()
    }
    
    func loadModel() {
        let config = MLModelConfiguration()
        
        do {
            // FIX: Using 'SiliconSentinel' to match your likely filename.
            // If this line is red, RENAME your .mlpackage file in Xcode to 'SiliconSentinel'
            let wrapper = try SystemMonitor(configuration: config)
            
            self.model = wrapper.model
            self.modelStatus = "SiliconSentinel Active"
            
            extractMetadata()
            
        } catch {
            self.modelStatus = "Error: \(error.localizedDescription)"
            print("Model Load Failed: \(error)")
        }
    }
    
    private func extractMetadata() {
        guard let model = model else { return }
        
        let metadata = model.modelDescription.metadata
        
        if let userDict = metadata[.creatorDefinedKey] as? [String: String] {
            
            if let meanStr = userDict["feature_means"],
               let stdStr = userDict["feature_stds"] {
                self.featureMeans = meanStr.split(separator: ",").compactMap { Float($0) }
                self.featureStds = stdStr.split(separator: ",").compactMap { Float($0) }
                print("✅ Metadata Loaded")
            }
            
            if let threshStr = userDict["suggested_threshold"],
               let threshVal = Double(threshStr) {
                self.alertThreshold = threshVal
                print("✅ Threshold Set to: \(self.alertThreshold)")
            }
        }
    }
    
    func analyze(_ raw: [Float]) {
        guard let model = model, !featureMeans.isEmpty else { return }
        
        let inputSize = featureMeans.count
        
        // FIX: The 'do' block must wrap ALL throwing calls (try)
        do {
            let inputArray = try MLMultiArray(shape: [1, NSNumber(value: inputSize)], dataType: .float32)
            
            for i in 0..<inputSize {
                if i < raw.count {
                    let std = (featureStds.count > i && featureStds[i] != 0) ? featureStds[i] : 1.0
                    let mean = (featureMeans.count > i) ? featureMeans[i] : 0.0
                    let scaled = (raw[i] - mean) / std
                    inputArray[i] = NSNumber(value: scaled)
                }
            }
            
            // Generic Prediction
            let inputFeatures = try MLDictionaryFeatureProvider(dictionary: ["input_features": inputArray])
            let output = try model.prediction(from: inputFeatures)
            
            // Extract Reconstruction
            guard let reconstruction = output.featureValue(for: "reconstruction")?.multiArrayValue else { return }
            
            var mse: Float = 0.0
            for i in 0..<inputSize {
                let val = inputArray[i].floatValue
                let recon = reconstruction[i].floatValue
                let diff = val - recon
                mse += (diff * diff)
            }
            
            DispatchQueue.main.async {
                self.currentScore = Double(mse)
                self.isAnomalous = self.currentScore > self.alertThreshold
                
                // TRIGGER THE DETECTIVE
                if self.isAnomalous {
                    DispatchQueue.global(qos: .userInitiated).async {
                        // Ensure ProcessInspector is available
                        let suspects = ProcessInspector.getTopProcesses()
                        DispatchQueue.main.async {
                            self.topSuspects = suspects
                        }
                    }
                } else {
                    self.topSuspects = []
                }
            }
            
        } catch {
            print("Inference Failed: \(error)")
        }
    }
}
