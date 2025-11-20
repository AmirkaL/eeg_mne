# Инструкция по деплою на GitHub

## Шаг 1: Создайте репозиторий на GitHub

1. Зайдите на https://github.com
2. Нажмите кнопку "+" в правом верхнем углу
3. Выберите "New repository"
4. Введите название репозитория (например: `eeg-go-nogo-processing`)
5. Выберите Public или Private
6. **НЕ** добавляйте README, .gitignore или license (они уже есть)
7. Нажмите "Create repository"

## Шаг 2: Подключите локальный репозиторий к GitHub

После создания репозитория GitHub покажет инструкции. Выполните:

```bash
git remote add origin https://github.com/ВАШ_USERNAME/НАЗВАНИЕ_РЕПОЗИТОРИЯ.git
git branch -M main
git push -u origin main
```

**Или если используете SSH:**

```bash
git remote add origin git@github.com:ВАШ_USERNAME/НАЗВАНИЕ_РЕПОЗИТОРИЯ.git
git branch -M main
git push -u origin main
```

## Шаг 3: Проверьте результат

Зайдите на страницу вашего репозитория на GitHub - все файлы должны быть там!

## Важно

- Данные участников (xlsx файлы) **не будут** загружены на GitHub (они в .gitignore)
- Результаты обработки (PNG, FIF, CSV) также **не будут** загружены
- На GitHub будут только скрипты, README и requirements.txt

## Если нужно обновить код на GitHub

После изменений в коде:

```bash
git add .
git commit -m "Описание изменений"
git push
```

