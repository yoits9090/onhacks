/* Dialogs — Evidence, Takeaway, How-it-works. Ported copy; dark instrument styling. */
import type { DerivedResults, EngineState, OffgridEngine } from '@/lib/engine'
import EvidenceContent from '@/components/EvidenceContent'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from '@/components/ui/dialog'
import { ScrollArea } from '@/components/ui/scroll-area'
import { ReticleMark } from '@/components/TopBar'

interface DialogsProps {
  engine: OffgridEngine
  results: DerivedResults
}

export default function Dialogs({ engine, results }: DialogsProps) {
  const state: EngineState = engine.state
  return (
    <>
      <Dialog open={state.evidenceOpen} onOpenChange={open => engine.setEvidenceOpen(open)}>
        <DialogContent className="max-h-[85dvh] w-full max-w-3xl gap-0 overflow-hidden border-border bg-[#0e141d] p-0 sm:max-w-3xl">
          <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
            <DialogTitle className="flex items-center gap-2 text-sm">
              <ReticleMark className="size-4 text-[#f0a828]" />
              Behind the estimate
            </DialogTitle>
          </div>
          <ScrollArea className="max-h-[calc(85dvh-49px)]">
            <div className="p-4">
              <DialogDescription className="sr-only">
                Measured evidence and limitations for the location estimate
              </DialogDescription>
              {state.evidenceStoppedWaiting && (
                <p className="mb-3 border border-[#f0a828]/40 bg-[#f0a828]/8 px-2.5 py-1.5 text-[11px] text-[#f0a828]">
                  Stopped waiting for inference. The server may still finish; no
                  unfinished edit comparison is shown.
                </p>
              )}
              <EvidenceContent state={state} results={results} />
            </div>
          </ScrollArea>
        </DialogContent>
      </Dialog>

      <Dialog open={state.takeawayOpen} onOpenChange={open => engine.setTakeawayOpen(open)}>
        <DialogContent className="max-w-lg gap-0 overflow-hidden border-border bg-[#0e141d] p-0 sm:max-w-lg">
          <div className="border-b border-border px-4 py-2.5">
            <span className="font-mono text-[10px] tracking-[0.28em] text-[#f0a828] uppercase">
              offgrid.
            </span>
          </div>
          <div className="space-y-3 p-4">
            <DialogTitle className="text-base">Less detail is not a guarantee.</DialogTitle>
            <DialogDescription className="sr-only">
              Closing notes on what editing a photo does and does not change
            </DialogDescription>
            <div className="space-y-2.5 text-xs leading-relaxed text-muted-foreground">
              <p>
                Removing metadata removes one source of location information.
                Visual clues can remain in the pixels.
              </p>
              <p>
                A moved estimate is a changed model answer—not proof of privacy,
                and not necessarily a more accurate answer.
              </p>
              <p>
                This is a street-photo study, not a GPS replacement. Inspect the
                evidence before generalizing from one photo.
              </p>
            </div>
            <Button
              className="mt-1 bg-[#f0a828] text-xs font-semibold text-[#14100a] hover:bg-[#f0a828]/85"
              onClick={() => engine.setTakeawayOpen(false)}
            >
              Back to the photo
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog open={state.howOpen} onOpenChange={open => engine.setHowOpen(open)}>
        <DialogContent className="max-w-lg gap-0 overflow-hidden border-border bg-[#0e141d] p-0 sm:max-w-lg">
          <div className="border-b border-border px-4 py-2.5">
            <span className="font-mono text-[10px] tracking-[0.28em] text-[#f0a828] uppercase">
              offgrid.
            </span>
          </div>
          <div className="space-y-3 p-4">
            <DialogTitle className="text-base">Look. Reveal. Rethink.</DialogTitle>
            <DialogDescription className="sr-only">
              How the Offgrid photo workflow works
            </DialogDescription>
            <ol className="space-y-3 text-xs leading-relaxed text-muted-foreground">
                <li>
                  <strong className="text-foreground">1 · Start with a photo.</strong>
                  <p>
                    Use a recorded street scene, or choose an image you have
                    permission to analyze. Original metadata is inspected locally;
                    GPS tags are not used for inference.
                  </p>
                </li>
                <li>
                  <strong className="text-foreground">2 · Reveal an estimate.</strong>
                  <p>
                    Examples show a saved result, clearly marked as recorded. Your
                    own photos require live inference. A separate reveal shows the
                    known location of an example.
                  </p>
                </li>
                <li>
                  <strong className="text-foreground">3 · Change what’s visible.</strong>
                  <p>
                    Cover a rectangular detail or apply a crop. With live
                    inference, test the edited pixels and compare estimates.
                    Without it, you can still edit and download a metadata-free
                    PNG.
                  </p>
                </li>
              </ol>
            <p className="text-[11px] text-muted-foreground/80">
              An estimate can be wrong. Removing metadata does not remove visual
              location clues.
            </p>
            <Button
              className="bg-[#f0a828] text-xs font-semibold text-[#14100a] hover:bg-[#f0a828]/85"
              onClick={() => engine.setHowOpen(false)}
            >
              Try a photo
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </>
  )
}
