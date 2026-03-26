//
//  MLMonitorApp.swift
//  CoreMetric
//
//  Created by Ege Kaya on 26.11.2025.
//

import SwiftUI

@main
struct CoreMetricApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        WindowGroup("CoreMetric") {
            ContentView()
                .environmentObject(appDelegate.engine)
                .frame(minWidth: 720, minHeight: 580)
        }
        .windowStyle(.titleBar)
        .windowToolbarStyle(.unified(showsTitle: true))
        .defaultSize(width: 880, height: 680)

        Settings {
            SettingsView()
        }
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    /// Single engine instance shared between the window and the menu bar icon.
    let engine = InferenceEngine()
    var menuBarManager: MenuBarManager?

    func applicationDidFinishLaunching(_ notification: Notification) {
        menuBarManager = MenuBarManager(engine: engine)
        NotificationManager.shared.requestPermission()
    }

    /// Don't quit when the window is closed — keep running via the menu bar icon.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    /// Clicking the Dock icon when no window is visible re-opens it.
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag {
            sender.windows.first { $0.canBecomeMain }?.makeKeyAndOrderFront(nil)
        }
        return true
    }
}
