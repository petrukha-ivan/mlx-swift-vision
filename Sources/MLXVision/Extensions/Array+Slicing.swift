//
//  Array+Slicing.swift
//  MLXVision
//
//  Created by Ivan Petrukha on 05.11.2025.
//

import Foundation

extension Array {

    subscript(_ indices: [Index]) -> [Element] {
        indices.map { self[$0] }
    }

    func indices(where predicate: (Element) -> Bool) -> [Index] {
        self.enumerated().filter({ predicate($0.element) }).map({ $0.offset })
    }
}

func zip<S1: Sequence, S2: Sequence, S3: Sequence>(
    _ s1: S1,
    _ s2: S2,
    _ s3: S3
) -> Array<(S1.Element, S2.Element, S3.Element)> {
    zip(s1, zip(s2, s3)).map {
        ($0.0, $0.1.0, $0.1.1)
    }
}

func zip<S1: Sequence, S2: Sequence, S3: Sequence, S4: Sequence>(
    _ s1: S1,
    _ s2: S2,
    _ s3: S3,
    _ s4: S4
) -> Array<(S1.Element, S2.Element, S3.Element, S4.Element)> {
    zip(s1, zip(s2, zip(s3, s4))).map {
        ($0.0, $0.1.0, $0.1.1.0, $0.1.1.1)
    }
}

func zip<S1: Sequence, S2: Sequence, S3: Sequence, S4: Sequence, S5: Sequence>(
    _ s1: S1,
    _ s2: S2,
    _ s3: S3,
    _ s4: S4,
    _ s5: S5
) -> Array<(S1.Element, S2.Element, S3.Element, S4.Element, S5.Element)> {
    zip(s1, zip(s2, zip(s3, zip(s4, s5)))).map {
        ($0.0, $0.1.0, $0.1.1.0, $0.1.1.1.0, $0.1.1.1.1)
    }
}

func zip<S1: Sequence, S2: Sequence, S3: Sequence, S4: Sequence, S5: Sequence, S6: Sequence>(
    _ s1: S1,
    _ s2: S2,
    _ s3: S3,
    _ s4: S4,
    _ s5: S5,
    _ s6: S6
) -> Array<(S1.Element, S2.Element, S3.Element, S4.Element, S5.Element, S6.Element)> {
    zip(s1, zip(s2, zip(s3, zip(s4, zip(s5, s6))))).map {
        ($0.0, $0.1.0, $0.1.1.0, $0.1.1.1.0, $0.1.1.1.1.0, $0.1.1.1.1.1)
    }
}
