import Foundation
import PDFKit
import CoreGraphics
import ImageIO
import CryptoKit

func pdfName(_ dictionary: CGPDFDictionaryRef, _ key: String) -> String {
    var pointer: UnsafePointer<CChar>?
    return CGPDFDictionaryGetName(dictionary, key, &pointer) && pointer != nil
        ? String(cString: pointer!) : ""
}

func embeddedFont(_ dictionary: CGPDFDictionaryRef) -> Bool {
    if pdfName(dictionary, "Subtype") == "Type3" { return true }
    var descendants: CGPDFArrayRef?
    if CGPDFDictionaryGetArray(dictionary, "DescendantFonts", &descendants), let array = descendants {
        for index in 0..<CGPDFArrayGetCount(array) {
            var child: CGPDFDictionaryRef?
            if !CGPDFArrayGetDictionary(array, index, &child) || child == nil || !embeddedFont(child!) { return false }
        }
        return true
    }
    var descriptor: CGPDFDictionaryRef?
    guard CGPDFDictionaryGetDictionary(dictionary, "FontDescriptor", &descriptor), let value = descriptor else { return false }
    for key in ["FontFile", "FontFile2", "FontFile3"] {
        var stream: CGPDFStreamRef?
        if CGPDFDictionaryGetStream(value, key, &stream) { return true }
    }
    return false
}

final class FontList {
    var entries: [[String: Any]] = []
}

func collectFont(_ key: UnsafePointer<CChar>, _ object: CGPDFObjectRef, _ info: UnsafeMutableRawPointer?) {
    guard let info = info else { return }
    let list = Unmanaged<FontList>.fromOpaque(info).takeUnretainedValue()
    var dictionary: CGPDFDictionaryRef?
    if CGPDFObjectGetValue(object, .dictionary, &dictionary), let value = dictionary {
        list.entries.append(["resource": String(cString: key), "name": pdfName(value, "BaseFont"),
                             "subtype": pdfName(value, "Subtype"), "embedded": embeddedFont(value)])
    }
}

let pdfURL = URL(fileURLWithPath: CommandLine.arguments[1])
let outdir = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
let document = PDFDocument(url: pdfURL)!
try FileManager.default.createDirectory(at: outdir, withIntermediateDirectories: true)
var records: [[String: Any]] = []
let fonts = FontList()
for index in 0..<document.pageCount {
    let page = document.page(at: index)!
    let bounds = page.bounds(for: .mediaBox)
    let scale: CGFloat = 2.0
    let context = CGContext(data: nil, width: Int(bounds.width * scale), height: Int(bounds.height * scale),
        bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    context.setFillColor(CGColor(gray: 1.0, alpha: 1.0))
    context.fill(CGRect(x: 0, y: 0, width: bounds.width * scale, height: bounds.height * scale))
    context.scaleBy(x: scale, y: scale)
    page.draw(with: .mediaBox, to: context)
    let name = String(format: "page-%02d.png", index + 1)
    let target = outdir.appendingPathComponent(name)
    let destination = CGImageDestinationCreateWithURL(target as CFURL, "public.png" as CFString, 1, nil)!
    CGImageDestinationAddImage(destination, context.makeImage()!, nil)
    guard CGImageDestinationFinalize(destination) else { fatalError("PNG write failed") }
    var textBounds = CGRect.null
    for character in 0..<page.numberOfCharacters {
        let box = page.characterBounds(at: character)
        if !box.isEmpty && !box.isNull && !box.isInfinite { textBounds = textBounds.union(box) }
    }
    if let cgpage = page.pageRef {
        var resources: CGPDFDictionaryRef?
        var dictionary: CGPDFDictionaryRef?
        if CGPDFDictionaryGetDictionary(cgpage.dictionary!, "Resources", &resources), let resource = resources,
           CGPDFDictionaryGetDictionary(resource, "Font", &dictionary), let fontDictionary = dictionary {
            CGPDFDictionaryApplyFunction(fontDictionary, collectFont, Unmanaged.passUnretained(fonts).toOpaque())
        }
    }
    records.append(["page": index + 1, "size_points": [bounds.width, bounds.height],
        "preview": target.path, "character_count": page.numberOfCharacters,
        "text_bounds_points": [textBounds.minX, textBounds.minY, textBounds.maxX, textBounds.maxY]])
    try (page.string ?? "").write(to: outdir.appendingPathComponent(String(format: "page-%02d.txt", index + 1)), atomically: true, encoding: .utf8)
}
var unique: [String: [String: Any]] = [:]
for font in fonts.entries { unique[font["name"] as! String] = font }
let pdfHash = SHA256.hash(data: try Data(contentsOf: pdfURL)).map { String(format: "%02x", $0) }.joined()
let report: [String: Any] = ["pdf": pdfURL.path, "pdf_sha256": pdfHash, "page_count": document.pageCount, "pages": records,
    "fonts": unique.keys.sorted().map { unique[$0]! },
    "all_fonts_embedded": !unique.isEmpty && unique.values.allSatisfy { $0["embedded"] as? Bool == true }]
let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
try data.write(to: outdir.appendingPathComponent("PDF_INSPECTION.json"))
print(String(data: data, encoding: .utf8)!)
