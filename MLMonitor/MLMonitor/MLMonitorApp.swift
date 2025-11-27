//
//  MLMonitorApp.swift
//  MLMonitor
//
//  Created by Ege Kaya on 26.11.2025.
//

import SwiftUI

@main
struct MLMonitorApp: App {
    // Connect the AppDelegate
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    
    var body: some Scene {
        Settings {
            // This ensures we can still have a Cmd+, Settings window later
            Text("Settings go here")
        }
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    var menuBarManager: MenuBarManager?
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Initialize the Menu Bar Manager
        menuBarManager = MenuBarManager()
    }
}
