"""将 basic_data.xlsx 按规模拆分为多个 CSV 文件。

Excel 中每个 sheet 是一个规模 (如 6x6, 10x5)，每个 sheet 内可能包含多个子问题
(如 6x6x3, 6x6x2 分别代表不同工人数量)。

表头行使用了合并单元格（如 "Machine" 跨 M 列，"W1,W2" 各跨 1 列），部分 sheet
的机器序列列与加工时间列之间还有间隙列（如 6x6x2 列 7 为空），不能简单用公式
`1+2M` 计算总列数。改为从数据行推断实际列跨度。
"""

import csv
import os
import re

import openpyxl


def parse_dimensions(name):
    m = re.match(r"^(\d+)x(\d+)x(\d+)$", name)
    if not m:
        raise ValueError(f"无法解析子问题名称: {name}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def make_columns(num_machines):
    cols = ["Job"]
    for i in range(1, num_machines + 1):
        cols.append(f"Machine_Op{i}")
    for i in range(1, num_machines + 1):
        cols.append(f"Time_Op{i}")
    return cols


def parse_sub_problems(ws):
    """解析 worksheet 中的所有子问题。"""
    sub_problems = []
    row_idx = 0
    all_rows = list(ws.iter_rows(min_row=1, max_row=ws.max_row, values_only=True))

    while row_idx < len(all_rows):
        row = all_rows[row_idx]
        first_val = row[0] if row else None
        if (
            first_val
            and isinstance(first_val, str)
            and re.match(r"^\d+x\d+x\d+$", first_val.strip())
        ):
            name = first_val.strip()
            num_jobs, num_machines, num_workers = parse_dimensions(name)
            row_idx += 1

            # 跳过元数据行直到表头行 (第 0 列 == "Job")，然后跳过该表头行
            while row_idx < len(all_rows):
                r = all_rows[row_idx]
                fv = (r[0] or "").strip() if r and r[0] else ""
                if fv == "Job":
                    row_idx += 1  # 跳过表头行
                    break
                if re.match(r"^\d+x\d+x\d+$", fv):
                    break  # 下一个子问题（异常）
                row_idx += 1

            # 收集数据行：提取每行非空值（自动跳过 Excel 中的间隙列）
            expected_cols = 1 + 2 * num_machines
            data_rows = []
            while row_idx < len(all_rows):
                r = all_rows[row_idx]
                fv = (r[0] or "").strip() if r and r[0] else ""

                if re.match(r"^\d+x\d+x\d+$", fv):
                    break
                if all(v is None for v in r):
                    row_idx += 1
                    break

                values = [str(v).strip() for v in r if v is not None]
                if len(values) >= expected_cols:
                    data_rows.append(values[:expected_cols])
                row_idx += 1

            sub_problems.append((name, num_jobs, num_machines, num_workers, data_rows))
            continue
        row_idx += 1

    return sub_problems


def main():
    src = os.path.join("data", "basic_data.xlsx")
    out_dir = os.path.join("data", "csv_output")
    os.makedirs(out_dir, exist_ok=True)

    wb = openpyxl.load_workbook(src)

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        sub_problems = parse_sub_problems(ws)

        if not sub_problems:
            print(f"[跳过] {sheet_name}: 未找到子问题")
            continue

        for name, num_jobs, num_machines, num_workers, data_rows in sub_problems:
            columns = make_columns(num_machines)
            if not data_rows:
                print(f"[跳过] {name}: 无数据行")
                continue

            out_path = os.path.join(out_dir, f"{name}.csv")
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                writer.writerows(data_rows)

            print(
                f"[完成] {name}.csv — {num_jobs}J × {num_machines}M × {num_workers}W, {len(data_rows)} 行"
            )

    wb.close()
    print(f"\n所有 CSV 文件已输出到 {out_dir}")


if __name__ == "__main__":
    main()
