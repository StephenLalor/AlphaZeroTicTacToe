// TODO: Highlight the current best node in some special way on the graph.

function visitsToColor(visits, maxVisits) {
    const ratio = visits / maxVisits
    const color_scale_max = Math.min(255, Math.floor(255 * ratio))
    const color_scale_min = Math.min(255, Math.floor(255 * (1 - ratio)))
    return `rgb(${color_scale_max}, 0, ${color_scale_min})`
}

function normMaxNodeVisits(treeData, percentile) {
    const nodeVisits = treeData.nodes.map((node) => node.label.visits)
    nodeVisits.sort((a, b) => a - b)
    const index = Math.ceil(percentile * nodeVisits.length) - 1
    return nodeVisits[index]
}

function getNodeHover(node) {
    const label = node.label
    const hoverData = `
        <div style="text-align: left;">
            <strong>Player:</strong> ${label.player}<br>
            <strong>Move:</strong> (${label.move})<br>
            <strong>Visits:</strong> ${label.visits}<br>
            <strong>Value:</strong> ${label.value}
        </div>`
    return hoverData
}

export function formatNodes(treeData) {
    const maxVisits = normMaxNodeVisits(treeData, 0.95)
    const transformedNodes = treeData.nodes.map((node) => {
        const nodeColor = visitsToColor(node.label.visits, maxVisits)
        return {
            id: node.id,
            board: node.label.board,
            label: '',
            title: getNodeHover(node),
            borderWidth: 3,
            color: {
                background: nodeColor,
                border: 'black',
                hover: { background: nodeColor, border: 'black' },
                highlight: { background: nodeColor, border: 'gray' },
            },
        }
    })
    return transformedNodes
}

export function formatEdges(treeData) {
    const transformedEdges = treeData.edges
        .filter((edge) => edge !== null)
        .map((edge) => ({
            from: edge.from,
            to: edge.to,
            width: 2,
            arrows: {
                to: {
                    enabled: true,
                    type: 'arrow',
                    scaleFactor: 0.75, // Arrow size.
                },
            },
        }))
    return transformedEdges
}

export function renderBoard(boardElement, boardData) {
    boardElement.innerHTML = '' // Clear any existing cells
    boardData.forEach((row, rowIndex) => {
        row.forEach((cell, colIndex) => {
            const cellElement = document.createElement('div')
            cellElement.className = 'cell'
            cellElement.textContent = cell
            boardElement.appendChild(cellElement)
        })
    })
}
