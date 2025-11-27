//
//  ProcessInspector.swift
//  MLMonitor
//
//  Created by Ege Kaya on 27.11.2025.
//

import Foundation

// FIX: Renamed from ProcessInfo to SuspectProcess to avoid conflict
struct SuspectProcess: Identifiable {
    let id = UUID()
    let pid: String
    let name: String
    let cpu: Double
}

class ProcessInspector {
    
    // FIX: Updated return type
    static func getTopProcesses() -> [SuspectProcess] {
        let task = Process()
        task.launchPath = "/bin/ps"
        task.arguments = ["-Ac", "-o", "%cpu,comm", "-r"]
        
        let pipe = Pipe()
        task.standardOutput = pipe
        
        do {
            try task.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            if let output = String(data: data, encoding: .utf8) {
                return parsePSOutput(output)
            }
        } catch {
            print("Error running ps: \(error)")
        }
        
        return []
    }
    
    // FIX: Updated return type
    private static func parsePSOutput(_ output: String) -> [SuspectProcess] {
        var results: [SuspectProcess] = []
        let lines = output.components(separatedBy: .newlines)
        
        for line in lines.dropFirst().prefix(3) {
            let parts = line.trimmingCharacters(in: .whitespaces)
                            .components(separatedBy: .whitespaces)
                            .filter { !$0.isEmpty }
            
            if parts.count >= 2 {
                if let cpu = Double(parts[0]) {
                    let name = parts.dropFirst().joined(separator: " ")
                    // FIX: Updated init
                    results.append(SuspectProcess(pid: "0", name: name, cpu: cpu))
                }
            }
        }
        return results
    }
}
