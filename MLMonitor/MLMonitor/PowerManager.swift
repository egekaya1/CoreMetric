//
//  PowerManager.swift
//  MLMonitor
//
//  Created by Ege Kaya on 27.11.2025.
//

import Foundation
import IOKit.ps
import Combine

class PowerManager: ObservableObject {
    @Published var isOnBattery: Bool = false
    @Published var thermalState: ProcessInfo.ThermalState = .nominal
    
    // Singleton for global access
    static let shared = PowerManager()
    
    private init() {
        // Initial check
        updatePowerStatus()
        
        // Monitor Thermal State (Don't run heavy ML if Mac is melting)
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(thermalStateChanged),
            name: ProcessInfo.thermalStateDidChangeNotification,
            object: nil
        )
        
        // Poll power source changes (IOKit loop)
        // A simple timer is cleaner than setting up a C-style RunLoopSource for this scale
        Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { _ in
            self.updatePowerStatus()
        }
    }
    
    @objc func thermalStateChanged() {
        DispatchQueue.main.async {
            self.thermalState = ProcessInfo.processInfo.thermalState
        }
    }
    
    func updatePowerStatus() {
        // C-API Call to get Power Source Info
        let snapshot = IOPSCopyPowerSourcesInfo().takeRetainedValue()
        let sources = IOPSCopyPowerSourcesList(snapshot).takeRetainedValue() as Array
        
        var onBattery = false
        
        for source in sources {
            if let info = IOPSGetPowerSourceDescription(snapshot, source).takeUnretainedValue() as? [String: Any] {
                if let state = info[kIOPSPowerSourceStateKey] as? String {
                    if state == kIOPSBatteryPowerValue {
                        onBattery = true
                    }
                }
            }
        }
        
        DispatchQueue.main.async {
            if self.isOnBattery != onBattery {
                self.isOnBattery = onBattery
                print("⚡️ Power Source Changed: \(onBattery ? "Battery 🔋" : "AC Power 🔌")")
            }
        }
    }
    
    // Recommended Interval based on state
    var recommendedInterval: TimeInterval {
        if thermalState == .critical { return 10.0 } // Cool down!
        if isOnBattery { return 5.0 } // Save battery
        return 1.0 // High Performance
    }
}
