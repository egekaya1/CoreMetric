//
//  NotificationManager.swift
//  CoreMetric
//
//  Created by Ege Kaya on 26.03.2026.
//

import UserNotifications
import SwiftUI

class NotificationManager {
    static let shared = NotificationManager()

    private init() {}

    func requestPermission() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { granted, error in
            if let error { print("Notification permission error: \(error)") }
        }
    }

    func sendAnomalyAlert(score: Double) {
        let enabled = UserDefaults.standard.object(forKey: "notificationsEnabled") as? Bool ?? true
        guard enabled else { return }

        let content = UNMutableNotificationContent()
        content.title = "Anomaly Detected"
        content.body  = "CoreMetric detected unusual system behavior (score: \(String(format: "%.3f", score)))"
        content.sound = .default

        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content: content,
            trigger: nil
        )
        UNUserNotificationCenter.current().add(request)
    }
}
