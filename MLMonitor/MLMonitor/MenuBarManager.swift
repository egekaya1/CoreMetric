//
//  MenuBarManager.swift
//  CoreMetric
//
//  Created by Ege Kaya on 27.11.2025.
//

import SwiftUI
import AppKit

class MenuBarManager: NSObject {
    private var statusItem: NSStatusItem!

    /// Shared engine — owned by AppDelegate, passed in on init.
    let engine: InferenceEngine

    init(engine: InferenceEngine) {
        self.engine = engine
        super.init()
        setupMenuBar()
    }

    private func setupMenuBar() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)

        if let button = statusItem.button {
            button.image = NSImage(systemSymbolName: "waveform.path.ecg",
                                   accessibilityDescription: "CoreMetric")
            button.action = #selector(showMainWindow)
            button.target = self
        }

        Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            self?.updateIconStatus()
        }
    }

    /// Bring the main window to front, or un-hide it if it was closed.
    @objc func showMainWindow() {
        NSApp.activate(ignoringOtherApps: true)
        if let window = NSApp.windows.first(where: { $0.canBecomeMain }) {
            window.makeKeyAndOrderFront(nil)
        }
    }

    func updateIconStatus() {
        DispatchQueue.main.async { [weak self] in
            guard let self, let button = self.statusItem.button else { return }
            let isAnomalous = self.engine.isAnomalous
            let symbol = isAnomalous ? "exclamationmark.triangle.fill" : "waveform.path.ecg"
            let color: NSColor = isAnomalous ? .red : .labelColor
            let config = NSImage.SymbolConfiguration(paletteColors: [color])
            button.image = NSImage(systemSymbolName: symbol, accessibilityDescription: nil)?
                .withSymbolConfiguration(config)
            let score = self.engine.currentScore
            button.title = score > 0.0001 ? " \(String(format: "%.3f", score))" : ""
        }
    }
}
