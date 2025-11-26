//
//  SystemCollector.swift
//  MLMonitor
//
//  Created by Ege Kaya on 26.11.2025.
//

import Foundation
import Darwin

class SystemCollector {
    // We store previous CPU ticks to calculate the delta (usage over time)
    private var prevLoad: host_cpu_load_info? = nil
    
    func getSystemMetrics() -> [Float] {
        return [
            getCPUUsage(),      // Feature 0: CPU %
            getMemoryUsage(),   // Feature 1: Memory %
            0.0,                // Feature 2: Disk Read
            0.0,                // Feature 3: Disk Write
            0.0,                // Feature 4: Net Sent
            0.0                 // Feature 5: Net Recv
        ]
    }
    
    // MARK: - CPU Magic
    private func getCPUUsage() -> Float {
        // FIX: Calculate the count manually instead of using the C macro
        var count = mach_msg_type_number_t(MemoryLayout<host_cpu_load_info_data_t>.size / MemoryLayout<integer_t>.size)
        var info = host_cpu_load_info()
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, $0, &count)
            }
        }
        
        guard result == KERN_SUCCESS else { return 0.0 }
        
        // If this is our first run, just store the baseline and return 0
        guard let prev = prevLoad else {
            prevLoad = info
            return 0.0
        }
        
        // Calculate Delta
        let user   = Float(info.cpu_ticks.0 - prev.cpu_ticks.0)
        let system = Float(info.cpu_ticks.1 - prev.cpu_ticks.1)
        let idle   = Float(info.cpu_ticks.2 - prev.cpu_ticks.2)
        let nice   = Float(info.cpu_ticks.3 - prev.cpu_ticks.3)
        
        let total = user + system + idle + nice
        prevLoad = info // Update baseline
        
        if total == 0 { return 0.0 }
        
        // Return usage fraction (0.0 to 1.0)
        return (user + system + nice) / total
    }
    
    // MARK: - RAM Magic
    private func getMemoryUsage() -> Float {
        // Calculate count manually here too for safety
        var size = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        var info = vm_statistics64_data_t()
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(size)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &size)
            }
        }
        
        guard result == KERN_SUCCESS else { return 0.0 }
        
        let pageSize = Float(vm_kernel_page_size)
        let active    = Float(info.active_count) * pageSize
        let wired     = Float(info.wire_count) * pageSize
        let compressed = Float(info.compressor_page_count) * pageSize
        
        let used = active + wired + compressed
        let total = Float(ProcessInfo.processInfo.physicalMemory)
        
        return (used / total) // Return 0.0 to 1.0 fraction
    }
}
