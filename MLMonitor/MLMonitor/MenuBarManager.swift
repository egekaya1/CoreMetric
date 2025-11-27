//
//  MenuBarManager.swift
//  MLMonitor
//
//  Created by Ege Kaya on 27.11.2025.
//

import SwiftUI
import AppKit

class MenuBarManager: NSObject {
    private var statusItem: NSStatusItem!
    private var popover: NSPopover!
    
    // We keep a reference to the Engine to pass it to the view
    private let engine = InferenceEngine()
    
    override init() {
        super.init()
        setupMenuBar()
        setupPopover()
    }
    
    private func setupMenuBar() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        
        if let button = statusItem.button {
            // Default Icon (SF Symbol)
            button.image = NSImage(systemSymbolName: "waveform.path.ecg", accessibilityDescription: "Monitor")
            button.action = #selector(togglePopover)
            button.target = self
        }
        
        // Timer to update the icon color based on Anomaly Score
        Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            self?.updateIconStatus()
        }
    }
    
    private func setupPopover() {
        popover = NSPopover()
        popover.behavior = .transient // Close when clicking outside
        popover.animates = true
        
        // Inject our existing ContentView
        // We force a specific size for the popover
        popover.contentViewController = NSHostingController(rootView:
            ContentView()
                .frame(width: 400, height: 600)
                .environmentObject(engine) // Pass engine if needed, or let View handle it
        )
    }
    
    @objc func togglePopover() {
        if let button = statusItem.button {
            if popover.isShown {
                popover.performClose(nil)
            } else {
                popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
                // Force app to front so the popover is interactive
                NSApp.activate(ignoringOtherApps: true)
            }
        }
    }
    
    func updateIconStatus() {
        // Change icon based on anomaly state
        DispatchQueue.main.async {
            if let button = self.statusItem.button {
                let symbol = self.engine.isAnomalous ? "exclamationmark.triangle.fill" : "waveform.path.ecg"
                let config = NSImage.SymbolConfiguration(paletteColors: [self.engine.isAnomalous ? .red : .labelColor])
                button.image = NSImage(systemSymbolName: symbol, accessibilityDescription: nil)?.withSymbolConfiguration(config)
            }
        }
    }
}
