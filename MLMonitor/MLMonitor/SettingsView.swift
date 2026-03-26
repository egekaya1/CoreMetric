//
//  SettingsView.swift
//  CoreMetric
//
//  Created by Ege Kaya on 26.03.2026.
//

import SwiftUI

struct SettingsView: View {
    @AppStorage("thresholdMultiplier")  private var multiplier: Double  = 1.0
    @AppStorage("alertCooldown")        private var cooldown: Double     = 5.0
    @AppStorage("notificationsEnabled") private var notifications: Bool  = true
    @AppStorage("autoStartMonitoring")  private var autoStart: Bool      = true

    var body: some View {
        Form {
            Section {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Sensitivity")
                        Spacer()
                        Text(sensitivityLabel)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Slider(value: $multiplier, in: 0.25...3.0, step: 0.25) {
                        EmptyView()
                    } minimumValueLabel: {
                        Text("High").font(.caption2)
                    } maximumValueLabel: {
                        Text("Low").font(.caption2)
                    }
                    Text("Threshold ×\(String(format: "%.2f", multiplier)): lower = more alerts")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Alert cooldown")
                        Spacer()
                        Text("\(Int(cooldown))s")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Slider(value: $cooldown, in: 1.0...60.0, step: 1.0)
                    Text("Minimum seconds between consecutive danger log entries")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            } header: {
                Text("Anomaly Detection")
            }

            Section {
                Toggle("Auto-start monitoring on open", isOn: $autoStart)
                Toggle("System notifications on anomaly", isOn: $notifications)
            } header: {
                Text("Behavior")
            }
        }
        .formStyle(.grouped)
        .frame(width: 400, height: 300)
    }

    private var sensitivityLabel: String {
        if multiplier <= 0.5  { return "Very High" }
        if multiplier <= 0.75 { return "High" }
        if multiplier <= 1.25 { return "Normal" }
        if multiplier <= 2.0  { return "Low" }
        return "Very Low"
    }
}

#Preview {
    SettingsView()
}
