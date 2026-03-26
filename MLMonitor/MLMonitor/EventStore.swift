//
//  EventStore.swift
//  CoreMetric
//
//  Created by Ege Kaya on 26.03.2026.
//

import Foundation
import Combine

class EventStore: ObservableObject {
    @Published private(set) var events: [LogEvent] = []

    private let fileURL: URL
    private let maxEvents = 500

    init() {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = support.appendingPathComponent("CoreMetric")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        fileURL = dir.appendingPathComponent("events.json")
        load()
    }

    func add(_ event: LogEvent) {
        events.insert(event, at: 0)
        if events.count > maxEvents {
            events = Array(events.prefix(maxEvents))
        }
        save()
    }

    func clear() {
        events = []
        save()
    }

    private func load() {
        guard let data = try? Data(contentsOf: fileURL),
              let decoded = try? JSONDecoder().decode([LogEvent].self, from: data) else { return }
        events = decoded
    }

    private func save() {
        guard let data = try? JSONEncoder().encode(events) else { return }
        try? data.write(to: fileURL, options: .atomic)
    }
}



