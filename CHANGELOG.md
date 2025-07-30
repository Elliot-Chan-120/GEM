7/28/2025 - GEM v1.0

Released GEM's first iteration :)

Future improvements:

* thorough error handling
* possible position weight matrix motif scanning
* enhancements to ReGen's logic
* DNA visualizations - linear gene map displaying sites of interest in comparison to the reference allele



7/30/2025 - GEM v1.1.0

ReGen algorithm enhancements

* stochastic mutation round now also includes random addition mutations instead of defaulting to a guided mutation
* this helps break through plateaus easier since plateaus are encountered during guided mutations anyway
* main now uses the correct filename

Benchmark gene 1 is now the first pathogenic gene variant in the CONTEXT\_ref\_alt\_df.pkl

* so it's an actual benchmark now not a pretend gene



7/30/2025 GEM v1.1.1

ReGen output fixes

* output txt file contains proper output, was previously outputting the reference under the label 'original variant'
* now in original variant stats, shows both ref and alt allele vcfs
* fixed benign threshold gene variants not all being represented
