using Documenter
using SafetySignalDetection

makedocs(
    sitename = "SafetySignalDetection",
    format = Documenter.HTML(),
    modules = [SafetySignalDetection]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "https://github.com/openpharma/SafetySignalDetection.jl.git"
)
