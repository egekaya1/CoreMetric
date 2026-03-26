//
//  MLMonitorApp.swift
//  CoreMetric
//
//  Created by Ege Kaya on 26.11.2025.
//

import SwiftUI

@main
struct CoreMetricApp: App {
    // Connect the AppDelegate
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    
    var body: some Scene {
        Settings {
            SettingsView()
        }
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    var menuBarManager: MenuBarManager?
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        menuBarManager = MenuBarManager()
        NotificationManager.shared.requestPermission()
    }
}
