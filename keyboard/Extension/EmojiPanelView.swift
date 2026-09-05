import UIKit

/// The native emoji keyboard's layout, measured on the same iPhone 17 (see
/// README): section title 11 pt below the top, five rows on a 38.75 pt row
/// pitch with the first row's centre 41.3 pt down, 45.8 pt columns starting
/// 3.4 pt in, an extra 14 pt between sections, sections filled column-major;
/// a category bar with its centre 236.3 pt down — "ABC" centred at x = 27.7,
/// nine category icons on a 33.3 pt pitch from x = 68, delete at x = W − 24.
/// The native search field is not reproduced (no offline emoji names).
final class EmojiPanelView: UIView, UICollectionViewDataSource, UICollectionViewDelegate {
    var onEmoji: ((String) -> Void)?
    var onBackspace: (() -> Void)?
    var onLetters: (() -> Void)?

    // measured geometry
    static let titleCenterY: CGFloat = 11.3
    static let firstRowCenterY: CGFloat = 41.3
    static let rowPitch: CGFloat = 38.75
    static let rows = 5
    static let columnPitch: CGFloat = 45.8
    static let gridLeft: CGFloat = 3.4
    static let sectionGap: CGFloat = 14
    static let barCenterY: CGFloat = 236.3
    static let abcCenterX: CGFloat = 27.7
    static let categoryStartX: CGFloat = 68
    static let categoryPitch: CGFloat = 33.3
    static let selectionDiameter: CGFloat = 29.3
    static let emojiFont = UIFont.systemFont(ofSize: 30)
    static let titleFont = UIFont.systemFont(ofSize: 13, weight: .semibold)

    private var sections: [(title: String, symbol: String, emoji: [String])] = []
    private let layout = EmojiLayout()
    private lazy var collection = UICollectionView(frame: .zero, collectionViewLayout: layout)
    private let abcButton = UIButton(type: .system)
    private let deleteButton = UIButton(type: .system)
    private var categoryButtons: [UIButton] = []
    private let selection = UIView()
    private var selected = 0

    override init(frame: CGRect) {
        super.init(frame: frame)
        reloadSections()
        collection.dataSource = self
        collection.delegate = self
        collection.backgroundColor = .clear
        collection.showsHorizontalScrollIndicator = false
        collection.alwaysBounceHorizontal = true
        collection.register(EmojiCell.self, forCellWithReuseIdentifier: "e")
        collection.register(TitleView.self, forSupplementaryViewOfKind: EmojiLayout.titleKind, withReuseIdentifier: "t")
        addSubview(collection)

        abcButton.setTitle("ABC", for: .normal)
        abcButton.titleLabel?.font = .systemFont(ofSize: 16, weight: .regular)
        abcButton.accessibilityIdentifier = "emoji.abc"
        abcButton.addTarget(self, action: #selector(lettersTapped), for: .touchUpInside)
        addSubview(abcButton)
        deleteButton.setImage(UIImage(systemName: "delete.left", withConfiguration: UIImage.SymbolConfiguration(pointSize: 22, weight: .regular)), for: .normal)
        deleteButton.accessibilityIdentifier = "emoji.delete"
        deleteButton.addTarget(self, action: #selector(deleteTapped), for: .touchUpInside)
        addSubview(deleteButton)

        selection.layer.cornerRadius = Self.selectionDiameter / 2
        addSubview(selection)
        for (i, s) in sections.enumerated() {
            let b = UIButton(type: .system)
            b.setImage(UIImage(systemName: s.symbol, withConfiguration: UIImage.SymbolConfiguration(pointSize: 17, weight: .regular)), for: .normal)
            b.tag = i
            b.accessibilityLabel = s.title.capitalized
            b.addTarget(self, action: #selector(categoryTapped(_:)), for: .touchUpInside)
            categoryButtons.append(b)
            addSubview(b)
        }
        restyle()
    }
    required init?(coder: NSCoder) { fatalError() }

    /// Recompute "Frequently Used" (call when the panel is shown).
    func reloadSections() {
        sections = [("FREQUENTLY USED", "clock", EmojiData.frequent())]
            + EmojiData.categories.map { ($0.title, $0.symbol, $0.emoji) }
        layout.sections = sections.map { $0.emoji.count }
        if collection.superview != nil { collection.reloadData() }
    }

    func restyle() {
        backgroundColor = Palette.background
        abcButton.setTitleColor(Palette.text, for: .normal)
        deleteButton.tintColor = Palette.text
        selection.backgroundColor = Palette.dark ? UIColor(white: 1, alpha: 0.18) : UIColor(white: 0, alpha: 0.1)
        for (i, b) in categoryButtons.enumerated() {
            b.tintColor = i == selected ? Palette.text : Palette.text.withAlphaComponent(0.55)
        }
        collection.reloadData()
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        let w = bounds.width
        let gridTop = Self.firstRowCenterY - Self.rowPitch / 2
        collection.frame = CGRect(x: 0, y: 0, width: w, height: gridTop + CGFloat(Self.rows) * Self.rowPitch)
        layout.topInset = gridTop
        abcButton.frame = CGRect(x: 0, y: Self.barCenterY - 20, width: 2 * Self.abcCenterX, height: 40)
        deleteButton.frame = CGRect(x: w - 24 - 30, y: Self.barCenterY - 20, width: 60, height: 40)
        for (i, b) in categoryButtons.enumerated() {
            let cx = Self.categoryStartX + CGFloat(i) * Self.categoryPitch
            b.frame = CGRect(x: cx - Self.categoryPitch / 2, y: Self.barCenterY - 20, width: Self.categoryPitch, height: 40)
        }
        moveSelection(animated: false)
    }

    private func moveSelection(animated: Bool) {
        let cx = Self.categoryStartX + CGFloat(selected) * Self.categoryPitch
        let f = CGRect(x: cx - Self.selectionDiameter / 2, y: Self.barCenterY - Self.selectionDiameter / 2,
                       width: Self.selectionDiameter, height: Self.selectionDiameter)
        if animated { UIView.animate(withDuration: 0.15) { self.selection.frame = f } } else { selection.frame = f }
        for (i, b) in categoryButtons.enumerated() {
            b.tintColor = i == selected ? Palette.text : Palette.text.withAlphaComponent(0.55)
        }
    }

    // MARK: actions

    @objc private func lettersTapped() { onLetters?() }
    @objc private func deleteTapped() { onBackspace?() }
    @objc private func categoryTapped(_ b: UIButton) {
        selected = b.tag
        moveSelection(animated: true)
        let x = min(layout.sectionX[b.tag], max(0, layout.collectionViewContentSize.width - collection.bounds.width))
        collection.setContentOffset(CGPoint(x: x, y: 0), animated: true)
    }

    func scrollViewDidScroll(_ scrollView: UIScrollView) {
        // The selected category follows the section at the left edge.
        let x = scrollView.contentOffset.x + 10
        var s = 0
        for (i, sx) in layout.sectionX.enumerated() where sx <= x { s = i }
        if s != selected { selected = s; moveSelection(animated: true) }
    }

    // MARK: data

    func numberOfSections(in collectionView: UICollectionView) -> Int { sections.count }
    func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int { sections[section].emoji.count }

    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
        let cell = collectionView.dequeueReusableCell(withReuseIdentifier: "e", for: indexPath) as! EmojiCell
        cell.label.text = sections[indexPath.section].emoji[indexPath.item]
        return cell
    }

    func collectionView(_ collectionView: UICollectionView, viewForSupplementaryElementOfKind kind: String, at indexPath: IndexPath) -> UICollectionReusableView {
        let v = collectionView.dequeueReusableSupplementaryView(ofKind: kind, withReuseIdentifier: "t", for: indexPath) as! TitleView
        v.label.text = sections[indexPath.section].title
        v.label.textColor = Palette.text.withAlphaComponent(0.45)
        return v
    }

    func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
        let e = sections[indexPath.section].emoji[indexPath.item]
        EmojiData.noteUse(e)
        onEmoji?(e)
    }

    final class EmojiCell: UICollectionViewCell {
        let label = UILabel()
        override init(frame: CGRect) {
            super.init(frame: frame)
            label.font = EmojiPanelView.emojiFont
            label.textAlignment = .center
            label.adjustsFontSizeToFitWidth = true
            contentView.addSubview(label)
        }
        required init?(coder: NSCoder) { fatalError() }
        override func layoutSubviews() { super.layoutSubviews(); label.frame = contentView.bounds }
        override var isHighlighted: Bool {
            didSet { contentView.backgroundColor = isHighlighted ? UIColor(white: 0.5, alpha: 0.2) : .clear
                     contentView.layer.cornerRadius = 8 }
        }
    }

    final class TitleView: UICollectionReusableView {
        let label = UILabel()
        override init(frame: CGRect) {
            super.init(frame: frame)
            label.font = EmojiPanelView.titleFont
            addSubview(label)
        }
        required init?(coder: NSCoder) { fatalError() }
        override func layoutSubviews() { super.layoutSubviews(); label.frame = bounds }
    }
}

/// Column-major horizontal layout: each section fills 5-row columns left to
/// right, a title sits above its first column, sections are separated by a gap.
final class EmojiLayout: UICollectionViewLayout {
    static let titleKind = "title"
    var sections: [Int] = [] { didSet { invalidateLayout() } }
    var topInset: CGFloat = 22 { didSet { invalidateLayout() } }
    private(set) var sectionX: [CGFloat] = []
    private var items: [[UICollectionViewLayoutAttributes]] = []
    private var titles: [UICollectionViewLayoutAttributes] = []
    private var width: CGFloat = 0

    override func prepare() {
        super.prepare()
        let p = EmojiPanelView.columnPitch, rp = EmojiPanelView.rowPitch, rows = EmojiPanelView.rows
        var x = EmojiPanelView.gridLeft
        items = []; titles = []; sectionX = []
        for (s, n) in sections.enumerated() {
            sectionX.append(max(0, x - EmojiPanelView.gridLeft))
            let cols = max(1, (n + rows - 1) / rows)
            let t = UICollectionViewLayoutAttributes(forSupplementaryViewOfKind: Self.titleKind, with: IndexPath(item: 0, section: s))
            t.frame = CGRect(x: x + 9.3, y: EmojiPanelView.titleCenterY - 9, width: CGFloat(cols) * p, height: 18)
            titles.append(t)
            var attrs: [UICollectionViewLayoutAttributes] = []
            for i in 0..<n {
                let a = UICollectionViewLayoutAttributes(forCellWith: IndexPath(item: i, section: s))
                a.frame = CGRect(x: x + CGFloat(i / rows) * p, y: topInset + CGFloat(i % rows) * rp, width: p, height: rp)
                attrs.append(a)
            }
            items.append(attrs)
            x += CGFloat(cols) * p + EmojiPanelView.sectionGap
        }
        width = x - EmojiPanelView.sectionGap + EmojiPanelView.gridLeft
    }

    override var collectionViewContentSize: CGSize {
        CGSize(width: max(width, collectionView?.bounds.width ?? 0), height: collectionView?.bounds.height ?? 0)
    }

    override func layoutAttributesForElements(in rect: CGRect) -> [UICollectionViewLayoutAttributes]? {
        var out = titles.filter { $0.frame.intersects(rect) }
        for s in items {
            guard let first = s.first, let last = s.last else { continue }
            if last.frame.maxX < rect.minX || first.frame.minX > rect.maxX { continue }
            out.append(contentsOf: s.filter { $0.frame.intersects(rect) })
        }
        return out
    }

    override func layoutAttributesForItem(at indexPath: IndexPath) -> UICollectionViewLayoutAttributes? {
        items[indexPath.section][indexPath.item]
    }

    override func layoutAttributesForSupplementaryView(ofKind elementKind: String, at indexPath: IndexPath) -> UICollectionViewLayoutAttributes? {
        titles[indexPath.section]
    }

    override func shouldInvalidateLayout(forBoundsChange newBounds: CGRect) -> Bool { false }
}
