//
//  AppSheet.swift
//  SimpleX (iOS)
//
//  Created by Evgeny on 24/11/2022.
//  Copyright © 2022 SimpleX Chat. All rights reserved.
//

import SwiftUI

class AppSheetState: ObservableObject {
    static let shared = AppSheetState()
    @Published var scenePhaseActive: Bool = false
    // Scehe phase is also be inactive while faceID is requested
    @Published var biometricAuth: Bool = false
}

private struct PrivacySensitive: ViewModifier {
    @AppStorage(DEFAULT_PRIVACY_PROTECT_SCREEN) private var protectScreen = false
    // Screen protection doesn't work for appSheet on iOS 16 if @Environment(\.scenePhase) is used instead of global state
    @ObservedObject var appSheetState: AppSheetState = AppSheetState.shared

    func body(content: Content) -> some View {
        if !protectScreen {
            content
        } else {
            content.privacySensitive(!appSheetState.scenePhaseActive && !appSheetState.biometricAuth).redacted(reason: .privacy)
        }
    }
}

extension View {
    func appSheet<Content>(
        isPresented: Binding<Bool>,
        onDismiss: (() -> Void)? = nil,
        content: @escaping () -> Content
    ) -> some View where Content: View {
        sheet(isPresented: isPresented, onDismiss: onDismiss) {
            content().modifier(PrivacySensitive())
        }
    }

    func appSheet<T, Content>(
        item: Binding<T?>,
        onDismiss: (() -> Void)? = nil,
        content: @escaping (T) -> Content
    ) -> some View where T: Identifiable, Content: View {
        sheet(item: item, onDismiss: onDismiss) { it in
            content(it).modifier(PrivacySensitive())
        }
    }
}
