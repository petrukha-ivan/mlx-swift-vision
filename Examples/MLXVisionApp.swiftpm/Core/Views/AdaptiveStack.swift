//
//  AdaptiveStack.swift
//  MLX Vision
//
//  Created by Ivan Petrukha on 22.02.2026.
//

import SwiftUI

struct AdaptiveStack<Header: View, Content: View, Footer: View>: View {

    @ViewBuilder let header: () -> Header
    @ViewBuilder let content: () -> Content
    @ViewBuilder let footer: () -> Footer

    var body: some View {
        #if targetEnvironment(macCatalyst)
            HStack {
                List {
                    header()
                    footer()
                }
                .containerRelativeFrame(.horizontal, count: 12, span: 4, spacing: 0)
                .scrollIndicators(.never)

                List {
                    content()
                }
                .containerRelativeFrame(.horizontal, count: 12, span: 8, spacing: 0)
                .scrollIndicators(.never)
            }
        #else
            List {
                header()
                content()
                footer()
            }
        #endif
    }
}
