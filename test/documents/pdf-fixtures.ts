import { createCanvas } from '@napi-rs/canvas';

function ascii(value: string): Uint8Array { return Uint8Array.from(value, (character) => character.charCodeAt(0)); }
function concatenate(chunks: Uint8Array[]): Uint8Array { const output = new Uint8Array(chunks.reduce((sum, chunk) => sum + chunk.length, 0)); let offset = 0; for (const chunk of chunks) { output.set(chunk, offset); offset += chunk.length; } return output; }
function stream(bytes: Uint8Array, dictionary = ''): Uint8Array { return concatenate([ascii(`<< /Length ${bytes.length} ${dictionary} >>\nstream\n`), bytes, ascii('\nendstream')]); }
function pdf(objects: (string | Uint8Array)[]): Uint8Array {
  const chunks = [ascii('%PDF-1.7\n')]; const offsets = [0]; let offset = chunks[0]!.length;
  objects.forEach((object, index) => { offsets.push(offset); const chunk = concatenate([ascii(`${index + 1} 0 obj\n`), typeof object === 'string' ? ascii(object) : object, ascii('\nendobj\n')]); chunks.push(chunk); offset += chunk.length; });
  chunks.push(ascii(`xref\n0 ${objects.length + 1}\n0000000000 65535 f \n${offsets.slice(1).map((value) => `${String(value).padStart(10, '0')} 00000 n \n`).join('')}trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${offset}\n%%EOF\n`));
  return concatenate(chunks);
}

/** Real two-page native-text PDF, including a geometric two-column/two-row table. */
export function nativePdfFixture(): Uint8Array {
  return pdf([
    '<< /Type /Catalog /Pages 2 0 R >>',
    '<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>',
    '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 300] /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>',
    '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 300] /Resources << /Font << /F1 5 0 R >> >> /Contents 7 0 R >>',
    '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
    stream(ascii('BT /F1 12 Tf 1 0 0 1 30 260 Tm (Inventory) Tj 1 0 0 1 30 220 Tm (Fruit) Tj 1 0 0 1 200 220 Tm (Count) Tj 1 0 0 1 30 200 Tm (Apples) Tj 1 0 0 1 200 200 Tm (7) Tj ET')),
    stream(ascii('BT /F1 12 Tf 1 0 0 1 30 260 Tm (Second page: native text.) Tj ET')),
  ]);
}

/** Actual raster glyphs, usable with Neko.ocr or as an embedded scan in a PDF. */
export function rasterOcrFixture(): { png: Uint8Array; jpeg: Uint8Array; width: number; height: number } {
  const width = 640; const height = 240; const canvas = createCanvas(width, height); const context = canvas.getContext('2d');
  context.fillStyle = 'white'; context.fillRect(0, 0, width, height); context.fillStyle = 'black'; context.font = '32px sans-serif'; context.fillText('Invoice 1042', 30, 60); context.fillText('Apples    7', 30, 120); context.fillText('Total     14', 30, 180);
  return { png: new Uint8Array(canvas.toBuffer('image/png')), jpeg: new Uint8Array(canvas.toBuffer('image/jpeg')), width, height };
}
export function scannedPdfFixture(): Uint8Array {
  const { jpeg, width, height } = rasterOcrFixture();
  return pdf([
    '<< /Type /Catalog /Pages 2 0 R >>', '<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
    '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 640 240] /Resources << /XObject << /Scan 4 0 R >> >> /Contents 5 0 R >>',
    stream(jpeg, `/Type /XObject /Subtype /Image /Width ${width} /Height ${height} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode`),
    stream(ascii('q 640 0 0 240 0 0 cm /Scan Do Q')),
  ]);
}
