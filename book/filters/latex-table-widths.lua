-- Give wrapping widths to tables whose source did not declare any.
--
-- Jupyter DataFrame outputs reach Pandoc with ColWidthDefault. In PDF that
-- becomes an l/l/l... longtable, so long prose columns can extend beyond the
-- page even though the HTML view is responsive. Explicit-width Markdown
-- tables are left untouched.

local function rows_from(section)
  if section == nil then
    return {}
  end
  return section.rows or section
end

local function update_lengths(lengths, rows)
  for _, row in ipairs(rows or {}) do
    for column_index, cell in ipairs(row.cells or {}) do
      local value = pandoc.utils.stringify(cell.contents or "")
      local length = utf8.len(value) or #value
      if length > lengths[column_index] then
        lengths[column_index] = length
      end
    end
  end
end

local function as_breakable_literal(value)
  -- xurl/nolinkurl handles slashes, dots, underscores and hyphens as valid
  -- breakpoints. Braces still delimit the macro argument and need escaping.
  local escaped = value:gsub("([{}])", "\\%1")
  return pandoc.RawInline("latex", "\\nolinkurl{" .. escaped .. "}")
end

local function looks_like_long_identifier(value)
  return #value >= 18
    and not value:find("%s")
    and (value:find("/", 1, true) ~= nil
      or value:find("_", 1, true) ~= nil
      or value:match("%.[%a%d][%a%d%-]*$") ~= nil)
end

function Table(table)
  if not FORMAT:match("latex") then
    return nil
  end

  local column_count = #table.colspecs
  if column_count < 2 then
    return nil
  end

  -- Inline code and path-like identifiers otherwise become unbreakable
  -- \texttt spans and can collide with the next column even after the table
  -- itself receives wrapping widths.
  table = table:walk({
    Code = function(code)
      return as_breakable_literal(code.text)
    end,
    Str = function(str)
      if looks_like_long_identifier(str.text) then
        return as_breakable_literal(str.text)
      end
      return nil
    end,
  })

  -- Preserve author-specified widths. The problematic DataFrame/pipe-table
  -- path has zero/default width in every column.
  for _, spec in ipairs(table.colspecs) do
    local width = spec[2]
    if type(width) == "number" and width > 0 then
      return table
    end
  end

  local lengths = {}
  for index = 1, column_count do
    lengths[index] = 0
  end

  update_lengths(lengths, rows_from(table.head))
  for _, body in ipairs(table.bodies or {}) do
    update_lengths(lengths, rows_from(body.head))
    update_lengths(lengths, rows_from(body.body))
  end
  update_lengths(lengths, rows_from(table.foot))

  -- Square-root weighting gives prose columns more room without letting one
  -- long cell consume the whole page. The floor keeps numeric/index columns
  -- readable; the cap prevents a paragraph from starving its neighbours.
  local weights = {}
  local total_weight = 0
  for index = 1, column_count do
    local bounded_length = math.max(6, math.min(lengths[index], 48))
    weights[index] = math.sqrt(bounded_length)
    total_weight = total_weight + weights[index]
  end

  for index, spec in ipairs(table.colspecs) do
    table.colspecs[index] = { spec[1], weights[index] / total_weight }
  end

  return table
end
